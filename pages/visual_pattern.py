from calendar import c
from ntpath import samefile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="데이터 시각화 대시보드", layout="wide")

if 'draw_graph' not in st.session_state:
    st.session_state['draw_graph'] = False

def load_csv_files_from_folder(folder_path):
    """폴더에서 모든 CSV 파일 로드"""
    csv_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        return []
    
    for file_path in folder.glob("*.csv"):
        try:
            df = pd.read_csv(file_path)
            
            # 컬럼명 정리: 앞뒤 공백 제거
            df.columns = df.columns.str.strip()
            
            # 빈 컬럼명이나 중복 컬럼명 처리
            df.columns = [col if col else f'unnamed_{i}' for i, col in enumerate(df.columns)]
            
            # 중복 컬럼명 처리
            cols = df.columns.tolist()
            seen = {}
            for i, col in enumerate(cols):
                if col in seen:
                    seen[col] += 1
                    cols[i] = f"{col}_{seen[col]}"
                else:
                    seen[col] = 0
            df.columns = cols
            
            csv_files.append({
                'filename': file_path.name,
                'filepath': str(file_path),
                'dataframe': df,
                'shape': df.shape
            })
        except Exception as e:
            st.warning(f"{file_path.name} 파일을 읽을 수 없습니다: {e}")
    
    return csv_files

def analyze_column_relationships(dataframes):
    """컬럼들 간의 관계성 분석"""
    all_columns = set()
    column_types = {}
    
    # 모든 데이터프레임에서 공통 컬럼 찾기
    for df_info in dataframes:
        df = df_info['dataframe']
        all_columns.update(df.columns)
        
        for col in df.columns:
            if col not in column_types:
                column_types[col] = []
            column_types[col].append(df[col].dtype)
    
    # 컬럼을 카테고리별로 분류
    timestamp_cols = []
    numeric_cols = []
    categorical_cols = []
    
    for col in all_columns:
        # 컬럼명 정리 후 분석
        col_clean = str(col).strip().lower()
        
        # 시간 관련 컬럼 식별
        if any(keyword in col_clean for keyword in ['timestamp', 'time', 'date',  '시간', '날짜']):
            timestamp_cols.append(col)
        # 숫자형 컬럼 식별
        elif any(pd.api.types.is_numeric_dtype(dtype) for dtype in column_types[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return {
        'timestamp_cols': timestamp_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'all_columns': list(all_columns)
    }

def group_columns_by_keywords(columns, keywords):
    """키워드별로 컬럼들을 그룹화"""
    groups = {}
    
    for keyword in keywords:
        keyword_lower = keyword.lower().strip()
        matching_cols = []
        
        for col in columns:
            # 컬럼명도 앞뒤 공백 제거 후 비교
            col_clean = str(col).strip().lower()
            if keyword_lower in col_clean:
                matching_cols.append(col)
        
        if matching_cols:
            groups[keyword] = matching_cols
    
    return groups

def create_keyword_aggregated_dataframe(dataframes, x_col, keyword_groups, agg_method='mean', time_agg_method='정확한 시간'):
    """키워드 그룹별로 집계된 데이터프레임 생성"""
    combined_data = []
    
    for df_info in dataframes:
        df = df_info['dataframe'].copy()
        df['source_file'] = df_info['filename']
        
        # X축 컬럼이 있는지 확인 (공백 제거 후)
        x_col_clean = str(x_col).strip()
        if x_col_clean not in df.columns:
            continue
        
        # 시간 컬럼 처리를 먼저 수행
        col_clean = str(x_col).strip().lower()
        if any(keyword in col_clean for keyword in ['time', 'date', 'timestamp']):
            try:
                df[x_col_clean] = pd.to_datetime(df[x_col_clean])
                
                # 시간 집계 방법에 따라 시간 컬럼 변환
                if time_agg_method == "시간별":
                    df['time_group'] = df[x_col_clean].dt.floor('H')  # 시간별로 그룹화
                elif time_agg_method == "일별":
                    df['time_group'] = df[x_col_clean].dt.date  # 일별로 그룹화
                elif time_agg_method == "월별":
                    df['time_group'] = df[x_col_clean].dt.to_period('M')  # 월별로 그룹화
                else:  # 정확한 시간
                    df['time_group'] = df[x_col_clean]
                    
            except Exception as e:
                st.warning(f"시간 컬럼 변환 실패: {e}")
                df['time_group'] = df[x_col_clean]
        else:
            df['time_group'] = df[x_col_clean]
            
        df_subset = df[['time_group', 'source_file']].copy()
        
        # 각 키워드 그룹별로 평균 계산
        for keyword, cols in keyword_groups.items():
            # 컬럼명 정리 후 사용 가능한 컬럼 찾기
            available_cols = []
            for col in cols:
                col_clean = str(col).strip()
                if col_clean in df.columns and pd.api.types.is_numeric_dtype(df[col_clean]):
                    available_cols.append(col_clean)
            
            if available_cols:
                try:
                    if agg_method == 'mean':
                        df_subset[f'{keyword}_평균'] = df[available_cols].mean(axis=1)
                    elif agg_method == 'sum':
                        df_subset[f'{keyword}_합계'] = df[available_cols].sum(axis=1)
                    elif agg_method == 'median':
                        df_subset[f'{keyword}_중앙값'] = df[available_cols].median(axis=1)
                    
                    # 디버깅 정보 추가
                    st.write(f"✅ '{keyword}' 그룹: {len(available_cols)}개 컬럼 집계됨")
                    with st.expander(f"'{keyword}' 그룹 상세"):
                        st.write(f"사용된 컬럼: {available_cols}")
                        
                except Exception as e:
                    st.warning(f"⚠️ '{keyword}' 그룹 집계 중 오류: {e}")
        
        combined_data.append(df_subset)
    
    if not combined_data:
        return None
    
    # 데이터 통합
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # time_group 기준으로 집계
    value_cols = [col for col in combined_df.columns if col not in ['time_group', 'source_file']]
    
    if not value_cols:
        st.error("집계할 수 있는 컬럼이 없습니다.")
        return None
    
    try:
        if agg_method == 'mean':
            agg_df = combined_df.groupby('time_group')[value_cols].mean().reset_index()
        elif agg_method == 'sum':
            agg_df = combined_df.groupby('time_group')[value_cols].sum().reset_index()
        elif agg_method == 'median':
            agg_df = combined_df.groupby('time_group')[value_cols].median().reset_index()
        else:
            agg_df = combined_df.groupby('time_group')[value_cols].mean().reset_index()
        
        # 원래 컬럼명으로 변경
        agg_df = agg_df.rename(columns={'time_group': x_col_clean})
        
        # 집계 결과 요약 표시
        st.info(f"📊 집계 완료: {len(agg_df)}개의 시간 구간, {len(value_cols)}개의 키워드 그룹")
        
        # 시간 범위 표시
        if len(agg_df) > 0:
            time_range_start = agg_df[x_col_clean].min()
            time_range_end = agg_df[x_col_clean].max()
            st.write(f"⏰ 분석 기간: {time_range_start} ~ {time_range_end}")
        
        return agg_df
    except Exception as e:
        st.error(f"데이터 집계 중 오류 발생: {e}")
        st.write(f"디버그 정보: 사용 가능한 컬럼 = {value_cols}")
        return None

def create_aggregated_dataframe(dataframes, group_cols, value_cols, agg_method='mean'):
    """여러 데이터프레임을 통합하여 집계된 데이터프레임 생성"""
    combined_data = []
    
    for df_info in dataframes:
        df = df_info['dataframe'].copy()
        df['source_file'] = df_info['filename']
        
        # 필요한 컬럼만 선택
        available_cols = [col for col in group_cols + value_cols if col in df.columns]
        if len(available_cols) < 2:
            continue
            
        df_subset = df[available_cols + ['source_file']]
        combined_data.append(df_subset)
    
    if not combined_data:
        return None
    
    # 데이터 통합
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # 시간 컬럼 처리
    for col in group_cols:
        if col in combined_df.columns:
            col_lower = col.lower().strip()
            if any(keyword in col_lower for keyword in ['timestamp', 'time', 'date', ]):
                try:
                    combined_df[col] = pd.to_datetime(combined_df[col])
                except:
                    pass
    
    # 집계 수행
    if agg_method == 'mean':
        agg_df = combined_df.groupby(group_cols)[value_cols].mean().reset_index()
    elif agg_method == 'sum':
        agg_df = combined_df.groupby(group_cols)[value_cols].sum().reset_index()
    elif agg_method == 'median':
        agg_df = combined_df.groupby(group_cols)[value_cols].median().reset_index()
    else:
        agg_df = combined_df.groupby(group_cols)[value_cols].mean().reset_index()
    
    return agg_df

def create_pattern_analysis_chart(df, x_col, y_cols, chart_type="Line"):
    """패턴 분석을 위한 차트 생성"""
    if chart_type == "Multi-Line with Correlation":
        # 서브플롯 생성 (상관관계 히트맵 포함)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('시계열 패턴', '상관관계 히트맵', '분포 분석', '트렌드 분석'),
            specs=[[{"colspan": 2}, None],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # 1. 시계열 패턴
        for y_col in y_cols:
            if y_col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df[x_col], y=df[y_col], name=y_col, mode='lines+markers'),
                    row=1, col=1
                )
        
        # 2. 상관관계 분석 (숫자형 컬럼만)
        numeric_cols = [col for col in y_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=1
            )
        
        # 3. 분포 분석 (박스플롯)
        for i, y_col in enumerate(numeric_cols[:3]):  # 최대 3개만
            fig.add_trace(
                go.Box(y=df[y_col], name=y_col),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="종합 패턴 분석")
        return fig
    
    elif chart_type == "Normalized Comparison":
        # 정규화된 비교 차트
        fig = go.Figure()
        
        # 각 컬럼을 0-1 범위로 정규화
        for y_col in y_cols:
            if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                normalized_values = (df[y_col] - df[y_col].min()) / (df[y_col].max() - df[y_col].min())
                fig.add_trace(
                    go.Scatter(
                        x=df[x_col], 
                        y=normalized_values, 
                        name=f"{y_col} (정규화)", 
                        mode='lines+markers'
                    )
                )
        
        fig.update_layout(
            title="정규화된 데이터 비교 (0-1 스케일)",
            yaxis_title="정규화된 값",
            height=600
        )
        return fig
    
    else:
        # 기본 차트들
        if chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_cols, title="라인 차트 - 패턴 분석")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_cols, title="산점도 - 패턴 분석")
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_cols, title="막대 차트 - 패턴 분석")
        
        return fig

# 사이드바
with st.sidebar:
    st.subheader("Server Info")
    st.markdown("**시흥 gpuserver2** (59.14.241.229) - 5090*3")
        
    st.title("⚙️ 설정")   

    # 데이터 소스 선택
    data_source = st.radio("데이터 소스 선택", ["단일 파일", "폴더 분석"])
    
    if data_source == "단일 파일":
        uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
        use_sample_data = st.checkbox("샘플 데이터 사용")
    else:
        folder_path = st.text_input("폴더 경로 입력", value="./sample")
        
    # 분석 모드 선택
    analysis_mode = st.radio("분석 모드", ["키워드 그룹 분석", "수동 선택"]) # "자동 패턴 분석" 일단 제거함
        
    if analysis_mode == "키워드 그룹 분석":
        st.subheader("🔍 키워드 설정")
        
        # 기본 키워드 제안
        default_keywords = st.text_input(
            "분석할 기본 키워드(쉼표로 구분)", 
            value="cell, temperature, current, soc"
        )
        keywords = [k.strip() for k in default_keywords.split(",") if k.strip()]
        
        # 추가 키워드
        additional_keywords = st.text_area(
            "추가 키워드 (한 줄에 하나씩)",
            placeholder="예:\speed\mileage\soh"
        )
        if additional_keywords:
            keywords.extend([k.strip() for k in additional_keywords.split("\n") if k.strip()])
    
    agg_method = st.selectbox("집계 방법", ["mean", "sum", "median"])
    
    chart_type = st.selectbox("차트 타입 선택", [
        "Line", "Scatter", "Bar", 
        "Multi-Line with Correlation", 
        "Normalized Comparison"
    ])
    show_table = st.checkbox("전체 데이터 테이블 보기")

# 메인 화면
st.title("📊 데이터 시각화 대시보드")
st.write("업로드한 데이터 또는 폴더 내 데이터를 다양한 차트로 시각화할 수 있습니다.")

# 폴더 분석 모드
if data_source == "폴더 분석":
    if folder_path:
        csv_files = load_csv_files_from_folder(folder_path)
else:
    csv_files = []
    df = pd.DataFrame()

    try:
        # 데이터 읽기
        if use_sample_data:
            filepath = "sample/628dani_V031BL0000_CASPER LONGRANGE_202410.csv"
            df = pd.read_csv(filepath)
            filename = "sampledata.csv"
        elif uploaded_file:
            df = pd.read_csv(uploaded_file)
            filepath = str(uploaded_file)
            filename = uploaded_file.name
        else:
            raise ValueError("파일 없음")

        # 컬럼명 정리
        df.columns = df.columns.str.strip()  # 앞뒤 공백 제거
        df.columns = [col if col else f'unnamed_{i}' for i, col in enumerate(df.columns)]  # 빈 컬럼 처리

        # 중복 컬럼명 처리
        seen = {}
        new_cols = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols

        # 결과 추가
        csv_files.append({
            'filename': filename,
            'filepath': filepath,
            'dataframe': df,
            'shape': df.shape
        })

    except Exception as e:
        st.warning("⚠️ CSV를 업로드하거나 샘플 데이터를 선택하세요.")
        
              

        
if csv_files:
    st.success(f"✅ {len(csv_files)}개의 CSV 파일을 발견했습니다.")
    
    # 파일 목록 표시
    with st.expander("발견된 파일들"):
        for file_info in csv_files:
            st.write(f"- **{file_info['filename']}**: {file_info['shape']} (행, 열)")
    
    # 컬럼 관계 분석
    column_analysis = analyze_column_relationships(csv_files)
    
    st.subheader("🔍 컬럼 분석 결과")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**시간 관련 컬럼**")
        for col in column_analysis['timestamp_cols']:
            st.write(f"- {col}")
    
    with col2:
        st.write("**숫자형 컬럼**")
        for col in column_analysis['numeric_cols'][:10]:  # 최대 10개만 표시
            st.write(f"- {col}")
        if len(column_analysis['numeric_cols']) > 10:
            st.write(f"... 외 {len(column_analysis['numeric_cols']) - 10}개")
    
    with col3:
        st.write("**범주형 컬럼**")
        for col in column_analysis['categorical_cols'][:10]:  # 최대 10개만 표시
            st.write(f"- {col}")
        if len(column_analysis['categorical_cols']) > 10:
            st.write(f"... 외 {len(column_analysis['categorical_cols']) - 10}개")
    
    # 분석 모드별 처리
    if analysis_mode == "키워드 그룹 분석":
        # 키워드 그룹 분석
        if keywords and column_analysis['timestamp_cols']:
            
            st.divider()
            st.subheader("🕒 시간 집계 설정")
            
            default_index = column_analysis['timestamp_cols'].index('timestamp') if 'timestamp' in column_analysis['timestamp_cols'] else 0
            
            x_col = st.selectbox("시간축 컬럼", column_analysis['timestamp_cols'], index=default_index)
            
            # 시간 집계 방법 선택 추가
            time_agg_method = st.selectbox(
                "시간 집계 방법", 
                ["정확한 시간", "시간별", "일별", "월별"],
                help="서로 다른 시간대의 데이터를 어떻게 집계할지 선택하세요"
            )
            
            if time_agg_method != "정확한 시간":
                st.info(f"💡 {time_agg_method} 집계: 같은 {time_agg_method.replace('별', '')} 내의 데이터들을 평균화합니다")
            
            # 키워드별 컬럼 그룹화
            keyword_groups = group_columns_by_keywords(column_analysis['numeric_cols'], keywords)
            
            st.divider()
            if keyword_groups:
                st.subheader("📊 키워드별 컬럼 그룹")
                
                # 각 키워드 그룹 표시
                for keyword, cols in keyword_groups.items():
                    with st.expander(f"🔹 '{keyword}' 관련 컬럼들 ({len(cols)}개)"):
                        for col in cols:
                            st.write(f"- {col}")
                
                # 분석 실행 버튼
                if st.button("🚀 키워드 그룹 분석 실행", type="primary"):
                    with st.spinner("키워드 그룹별 데이터를 분석하고 있습니다..."):
                        aggregated_df = create_keyword_aggregated_dataframe(
                            csv_files, x_col, keyword_groups, agg_method, time_agg_method
                        )
                    
                    if aggregated_df is not None and len(aggregated_df) > 0:
                        st.subheader("📈 키워드 그룹별 패턴 분석")
                        
                        # 요약 통계
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("분석된 키워드 그룹", len(keyword_groups))
                        with col2:
                            st.metric("데이터 포인트", len(aggregated_df))
                        with col3:
                            group_cols = [col for col in aggregated_df.columns if col != x_col]
                            st.metric("생성된 지표", len(group_cols))
                        
                        # 키워드 그룹별 차트
                        y_cols = [col for col in aggregated_df.columns if col != x_col]
                        
                        if chart_type == "Multi-Line with Correlation":
                            # 서브플롯 생성
                            fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=('키워드 그룹별 트렌드', '그룹간 상관관계', '분포 비교', '정규화 비교'),
                                specs=[[{"colspan": 2}, None],
                                        [{"type": "xy"}, {"type": "xy"}]]
                            )
                            
                            # 1. 키워드 그룹별 트렌드
                            for y_col in y_cols:
                                fig.add_trace(
                                    go.Scatter(x=aggregated_df[x_col], y=aggregated_df[y_col], 
                                                name=y_col, mode='lines+markers'),
                                    row=1, col=1
                                )
                            
                            # 2. 상관관계 히트맵
                            if len(y_cols) > 1:
                                corr_matrix = aggregated_df[y_cols].corr()
                                fig.add_trace(
                                    go.Heatmap(
                                        z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.columns,
                                        colorscale='RdBu',
                                        zmid=0,
                                        showscale=True
                                    ),
                                    row=2, col=1
                                )
                            
                            # 3. 정규화 비교
                            for y_col in y_cols:
                                if pd.api.types.is_numeric_dtype(aggregated_df[y_col]):
                                    col_data = aggregated_df[y_col]
                                    normalized = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                                    fig.add_trace(
                                        go.Scatter(x=aggregated_df[x_col], y=normalized, 
                                                    name=f"{y_col} (정규화)", mode='lines'),
                                        row=2, col=2
                                    )
                            
                            fig.update_layout(height=800, title_text="키워드 그룹별 종합 분석")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            # 기본 차트
                            fig = create_pattern_analysis_chart(aggregated_df, x_col, y_cols, chart_type)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 상세 통계 정보
                        with st.expander("📋 상세 통계 정보"):
                            st.write("**각 키워드 그룹별 기본 통계:**")
                            for col in y_cols:
                                col_stats = aggregated_df[col].describe()
                                st.write(f"**{col}**")
                                st.write(f"- 평균: {col_stats['mean']:.2f}")
                                st.write(f"- 표준편차: {col_stats['std']:.2f}")
                                st.write(f"- 최솟값: {col_stats['min']:.2f}")
                                st.write(f"- 최댓값: {col_stats['max']:.2f}")
                                st.write("---")
                        
                        # 데이터 테이블 (옵션)
                        if show_table:
                            st.subheader("집계된 키워드 그룹 데이터")
                            st.dataframe(aggregated_df, use_container_width=True)
                            
                            # CSV 다운로드 버튼
                            csv = aggregated_df.to_csv(index=False)
                            st.download_button(
                                label="키워드 그룹 분석 결과 다운로드 (CSV)",
                                data=csv,
                                file_name=f"keyword_group_analysis_{agg_method}.csv",
                                mime="text/csv"
                            )
                    else:
                        st.warning("⚠️ 분석할 수 있는 데이터가 없습니다. 키워드나 파일을 확인해주세요.")
            else:
                st.warning("⚠️ 지정된 키워드와 일치하는 컬럼을 찾을 수 없습니다.")
        else:
            st.info("💡 분석할 키워드를 입력하고 시간축 컬럼을 선택해주세요.")
    
    elif analysis_mode == "자동 패턴 분석":
        # 기존 자동 분석 로직
        if column_analysis['timestamp_cols'] and column_analysis['numeric_cols']:
            
            default_index = column_analysis['timestamp_cols'].index('timestamp') if 'timestamp' in column_analysis['timestamp_cols'] else 0
            
            x_col = column_analysis['timestamp_cols'][default_index]
            y_cols = column_analysis['numeric_cols'][:5]
            
            st.success(f"🤖 자동 선택: X축={x_col}, Y축={', '.join(y_cols)}")
            
            with st.spinner("데이터를 집계하고 있습니다..."):
                aggregated_df = create_aggregated_dataframe(csv_files, [x_col], y_cols, agg_method)
            
            if aggregated_df is not None:
                st.subheader("📈 패턴 분석 결과")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 데이터 포인트", len(aggregated_df))
                with col2:
                    st.metric("분석 기간", f"{len(aggregated_df)}개 구간")
                with col3:
                    st.metric("분석 지표", len(y_cols))
                
                fig = create_pattern_analysis_chart(aggregated_df, x_col, y_cols, chart_type)
                st.plotly_chart(fig, use_container_width=True)
                
                if show_table:
                    st.subheader("집계된 데이터")
                    st.dataframe(aggregated_df, use_container_width=True)
                    
                    csv = aggregated_df.to_csv(index=False)
                    st.download_button(
                        label="집계된 데이터 다운로드 (CSV)",
                        data=csv,
                        file_name=f"aggregated_data_{agg_method}.csv",
                        mime="text/csv"
                    )
        
    else:  # 수동 선택 모드
        st.subheader("수동 컬럼 선택")
        
        available_cols = column_analysis['all_columns']
        x_col = st.selectbox("X축 컬럼", available_cols)
        y_cols = st.multiselect("Y축 컬럼(복수 선택 가능)", available_cols)
        
        if x_col and y_cols and st.button("패턴 분석 실행"):
            with st.spinner("데이터를 분석하고 있습니다..."):
                aggregated_df = create_aggregated_dataframe(csv_files, [x_col], y_cols, agg_method)
            
            if aggregated_df is not None:
                fig = create_pattern_analysis_chart(aggregated_df, x_col, y_cols, chart_type)
                st.plotly_chart(fig, use_container_width=True)
                
                if show_table:
                    st.dataframe(aggregated_df, use_container_width=True)
else:
    if data_source == "폴더 분석":
        st.warning("⚠️ 지정된 폴더에서 CSV 파일을 찾을 수 없습니다.")


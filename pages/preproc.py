import streamlit as st
import pandas as pd
import numpy as np
import os
import io

# --- 기본 전처리 규칙 ---
def get_default_rules():
    return [
        {'pattern': 'soc', 'min': 0.0, 'max': 100.0},
        {'pattern': 'soh', 'min': 0.0, 'max': 100.0},
        {'pattern': 'pack_v', 'min': 0.0, 'max': 3000.0},
        {'pattern': 'temperature', 'min': -35.0, 'max': 80.0},
        {'pattern': 'curr', 'min': -500.0, 'max': 500.0},
        {'pattern': 'cell', 'min': -500.0, 'max': 500.0},
        {'pattern': 'speed', 'min': 0.0, 'max': 180.0},
        {'pattern': 'mileage', 'min': 0.0, 'max': 2000000.0},
    ]

# 규칙 관리: 세션 상태에 규칙 리스트 저장
if 'rules' not in st.session_state:
    st.session_state['rules'] = get_default_rules()
    
# 전처리된 결과를 임시 저장할 공간 확보
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = {}
    
# 사이드바
with st.sidebar:
    st.subheader("Server Info")
    st.markdown("**시흥 gpuserver2** (59.14.241.229) - 5090*3")
        
    st.title("⚙️ 설정")   
    uploaded_files = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"],accept_multiple_files=True)
    
    preview_button = st.button("미리보기")

st.title("배터와이 데이터 전처리")


st.subheader("설정된 규칙")
# 규칙 리스트 표시 및 삭제/수정
for i, rule in enumerate(st.session_state['rules']):
    with st.expander(f"규칙 {i+1}: {rule['pattern']} ∈ [{rule['min']}, {rule['max']}]", expanded=False):
        new_pattern = st.text_input(f"컬럼명 패턴_{i}", value=rule['pattern'], key=f"pattern_{i}")
        new_min = st.number_input(f"최소값_{i}", value=rule['min'], key=f"min_{i}")
        new_max = st.number_input(f"최대값_{i}", value=rule['max'], key=f"max_{i}")
        if st.button(f"규칙 수정", key=f"edit_{i}"):
            st.session_state['rules'][i] = {'pattern': new_pattern, 'min': new_min, 'max': new_max}
            st.rerun()
        if st.button(f"규칙 삭제", key=f"del_{i}"):
            st.session_state['rules'].pop(i)
            st.rerun()

st.subheader("전처리 규칙 관리")
with st.form("add_rule_form", clear_on_submit=True):
    col_pattern = st.text_input("컬럼명에 포함되는 문자열 (예: soc, pack_v, temperature 등)")
    min_val = st.number_input("최소값", value=0.0, format="%.4f")
    max_val = st.number_input("최대값", value=100.0, format="%.4f")
    submitted = st.form_submit_button("규칙 추가")
    if submitted and col_pattern:
        st.session_state['rules'].append({'pattern': col_pattern, 'min': min_val, 'max': max_val})

# st.subheader("전처리 파일 경로")
# folder = st.text_input("CSV 폴더 경로", value="./data")

# def list_csv_files_local(folder):
#     return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

def get_violation_counts_custom(df, rules):
    result = {}
    for rule in rules:
        pattern = rule['pattern']
        min_v = rule['min']
        max_v = rule['max']

        for col in [c for c in df.columns if pattern in c]:
            col_series = pd.to_numeric(df[col], errors='coerce')  # 문자열 등은 NaN으로 처리
            violation_mask = (col_series.notna()) & ((col_series < min_v) | (col_series > max_v))
            result[f"{col} ({min_v}~{max_v} 위반)"] = violation_mask.sum()  # ✅ 안쪽으로 들여쓰기
    return result

def custom_preprocessing(df, rules):
    for rule in rules:
        pattern = rule['pattern']
        min_v = rule['min']
        max_v = rule['max']

        for col in [c for c in df.columns if pattern in c]:
            # 문자열 포함 가능성을 고려하여 안전하게 숫자형으로 변환
            numeric_series = pd.to_numeric(df[col], errors='coerce')  # 문자열 → NaN
            # 범위 위반한 값도 NaN 처리
            mask = (numeric_series < min_v) | (numeric_series > max_v)
            numeric_series = numeric_series.mask(mask, np.nan)
            df[col] = numeric_series  # 원본에 덮어쓰기
    return df

if uploaded_files is not None and preview_button:
    csv_files = uploaded_files
    stats = []
    with st.spinner("규칙 위반 row 집계 중..."):
        for f in csv_files:
            df = pd.read_csv(f)
            violations = get_violation_counts_custom(df, st.session_state['rules'])
            violations['file'] = f.name +"_preproc"
            stats.append(violations)
    st.write("파일별 전처리 규칙 위반 row 수")
    st.dataframe(pd.DataFrame(stats).fillna(0))

# 전처리만 먼저 수행
if st.button("전처리 시작") :
    csv_files = uploaded_files
    st.session_state['processed_files'] = {}  # 기존 결과 초기화
    progress = st.progress(0)

    for i, f in enumerate(csv_files):
        df = pd.read_csv(f)
        df_proc = custom_preprocessing(df, st.session_state['rules'])

        # 결과 저장: dict에 파일명과 전처리 DataFrame 보관
        st.session_state['processed_files'][f.name] = df_proc

        progress.progress((i + 1) / len(csv_files))

    st.success("✅ 전처리 완료! '파일 다운로드' 버튼을 눌러 저장할 수 있습니다.")

# 파일 다운로드
if st.button("파일 다운로드"):
        
    if not st.session_state.get('processed_files'):
        st.warning("⚠️ 먼저 '전처리 시작'을 클릭하세요.")
    else:
        for filename, df_proc in st.session_state['processed_files'].items():
            buffer = io.StringIO()
            df_proc.to_csv(buffer, index=False)
            csv_str = buffer.getvalue()  # 문자열로 변환

            st.download_button(
                label=f"📥 {filename} 전처리 결과 다운로드",
                data=csv_str,  # 문자열 전달
                file_name=f"{os.path.splitext(filename)[0]}_preproc.csv",
                mime="text/csv"
            )

        st.success("💾 다운로드 준비 완료!")
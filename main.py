import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="데이터 기본 시각화 대시보드", layout="wide")

if 'draw_graph' not in st.session_state:
    st.session_state['draw_graph'] = False

# 사이드바
with st.sidebar:
    st.subheader("Server Info")
    st.markdown("**시흥 gpuserver2** (59.14.241.229) - 5090*3")
        
    st.title("⚙️ 설정")   

    uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
    use_sample_data = st.sidebar.checkbox("샘플 데이터 사용")
    chart_type = st.sidebar.selectbox("차트 타입 선택", [ "Line", "Scatter","Bar"])
    show_table = st.sidebar.checkbox("전체 데이터 테이블 보기")

# 메인 화면
st.title("📊 데이터 기본 시각화 대시보드")
st.write("해당 프로젝트는 배터와이 데이터에 최적화 되어 있습니다. 그 외 데이터는 적합하지 않을 수 있습니다.")

if uploaded_file or use_sample_data:
    if 'preview' not in st.session_state:
        st.session_state['preview'] = False
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("sample/628dani_V031BL0000_CASPER LONGRANGE_202410.csv")
        st.success("✅ 샘플 데이터를 사용하고 있습니다.")
    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    x_col = st.sidebar.selectbox("X축 컬럼", df.columns)

    # x축 범위 지정 UI
    x_data = df[x_col]
    x_dtype = x_data.dtype

    # 날짜형식 문자열 자동 인식 및 변환
    is_timestamp_col = (
        x_col.strip().lower() == 'timestamp' or
        (x_dtype == object and x_data.str.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?").all())
    ) 
    
    if is_timestamp_col:
        x_data = pd.to_datetime(x_data)
        x_dtype = x_data.dtype

    x_range = None
    if pd.api.types.is_numeric_dtype(x_dtype):
        min_val, max_val = float(x_data.min()), float(x_data.max())
        x_range = st.sidebar.slider("X축 범위", min_val, max_val, (min_val, max_val))
        mask = (x_data >= x_range[0]) & (x_data <= x_range[1])
    elif pd.api.types.is_datetime64_any_dtype(x_dtype):
        min_date, max_date = x_data.min().date(), x_data.max().date()
        x_range = st.sidebar.slider("X축 날짜 범위", min_date, max_date, (min_date, max_date))
        mask = (x_data.dt.date >= x_range[0]) & (x_data.dt.date <= x_range[1])
    else:
        unique_vals = x_data.unique()
        selected = st.sidebar.multiselect("X축 값 선택", unique_vals, default=list(unique_vals))
        mask = x_data.isin(selected)

    filtered_df = df[mask]

    # y축 컬럼 선택 및 미리보기
    y_cols = st.sidebar.multiselect("Y축 컬럼(복수 선택 가능)", df.columns)
    
    if y_cols and st.sidebar.button("미리보기"):
        st.session_state['preview'] = True
        st.session_state['draw_graph'] = False  # 그래프 상태 초기화

    # preview 상태이면 데이터 미리보기 보여주고 "그래프 그리기" 버튼 표시
    if st.session_state['preview'] and y_cols:
        st.subheader("미리보기 데이터")
        st.dataframe(filtered_df[[x_col] + y_cols], use_container_width=True)

        st.session_state['draw_graph'] = True
        
    if st.session_state['draw_graph'] and y_cols:
        if st.button("그래프 그리기"):
            with st.spinner("그래프를 생성하고 있습니다..."):
                st.subheader(f"{chart_type} 차트")

                # Plotly 차트 통합
                if chart_type == "Line":
                    fig = px.line(filtered_df, x=x_col, y=y_cols, title="라인 차트")
                elif chart_type == "Scatter":
                    fig = px.scatter(filtered_df, x=x_col, y=y_cols, title="산점도")
                elif chart_type == "Bar":
                    fig = px.bar(filtered_df, x=x_col, y=y_cols, title="막대 차트")

                st.plotly_chart(fig, use_container_width=True)
            
            # 다음 클릭 전까지는 다시 안보이도록
            st.session_state['draw_graph'] = False

else:
    st.info("좌측 사이드바에서 CSV 파일을 업로드하세요.")
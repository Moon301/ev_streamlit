import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="데이터 시각화 대시보드", layout="wide")

# 사이드바
with st.sidebar:
    st.subheader("Server Info")
    st.markdown("**시흥 gpuserver2** (59.14.241.229) - 5090*3")
        
    st.title("⚙️ 설정")   

    uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
    chart_type = st.sidebar.selectbox("차트 타입 선택", [ "Line", "Scatter","Bar"])
    show_table = st.sidebar.checkbox("전체 데이터 테이블 보기")

# 메인 화면
st.title("📊 데이터 시각화 대시보드")
st.write("업로드한 데이터를 다양한 차트로 시각화할 수 있습니다.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    x_col = st.sidebar.selectbox("X축 컬럼", df.columns)

    # x축 범위 지정 UI
    x_data = df[x_col]
    x_dtype = x_data.dtype

    # 날짜형식 문자열 자동 인식 및 변환
    is_timestamp_col = (
        x_col.lower() == 'timestamp' or
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
    preview = False
    if y_cols:
        if st.sidebar.button("미리보기"):
            preview = True
            st.session_state['preview'] = True
    if 'preview' not in st.session_state:
        st.session_state['preview'] = False
    if preview:
        st.session_state['preview'] = True
    # 미리보기가 된 상태에서만 그래프 그리기 버튼 생성
    if st.session_state['preview'] and y_cols:
        st.subheader("미리보기 데이터")
        st.dataframe(filtered_df[[x_col] + y_cols])
        if st.button("그래프 그리기"):
            with st.spinner("그래프를 생성하고 있습니다..."):
                st.subheader(f"{chart_type} 차트")
                fig, ax = plt.subplots(figsize=(6, 3))
                if chart_type == "Bar":
                    for y_col in y_cols:
                        ax.bar(filtered_df[x_col], filtered_df[y_col], label=y_col)
                elif chart_type == "Line":
                    for y_col in y_cols:
                        ax.plot(filtered_df[x_col], filtered_df[y_col], label=y_col)
                elif chart_type == "Scatter":
                    for y_col in y_cols:
                        ax.scatter(filtered_df[x_col], filtered_df[y_col], label=y_col)
                if y_cols:
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                st.pyplot(fig)

    if show_table:
        st.subheader("전체 데이터")
        st.dataframe(df)
else:
    st.info("좌측 사이드바에서 CSV 파일을 업로드하세요.")
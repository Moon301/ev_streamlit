import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


# 페이지 설정
st.set_page_config(
    page_title="배터와이 데이터 분석 대시보드",
    page_icon="🔋",
    layout="wide"
)

# 제목
st.title("🔋 배터와이 데이터 분석 대시보드")
st.markdown("---")

# 분석 결과 섹션
st.subheader("📈 데이터 요약")
st.write("**데이터 기간:** 2023년 11월 ~ 2025년 7월 (약 20개월)")

# 메트릭 카드들
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="총 행 수",
        value="583,414,240",
        help="전체 데이터의 총 행 개수"
    )

with col2:
    st.metric(
        label="파일 수량",
        value="12,860",
        help="처리된 CSV 파일의 총 개수"
    )

with col3:
    st.metric(
        label="최대 컬럼 수",
        value="244",
        help="단일 파일에서 발견된 최대 컬럼 개수"
    )

with col4:
    st.metric(
        label="고유 Client ID",
        value="400",
        help="데이터에서 발견된 고유한 클라이언트 ID 수"
    )

st.markdown("---")

# 상세 정보 섹션
with st.expander("📊 상세 분석 정보", expanded=True):
    st.subheader("데이터 규모")
    data_info = {
        "항목": ["총 행 수", "파일 수량", "최대 컬럼 수", "고유 Client ID"],
        "값": ["583,414,240", "12,860", "244", "400"],
        "설명": [
            "전체 데이터셋의 총 레코드 수",
            "분석된 CSV 파일의 개수", 
            "가장 많은 컬럼을 가진 파일의 컬럼 수",
            "데이터에 포함된 고유한 클라이언트 수"
        ]
    }
    
    df_info = pd.DataFrame(data_info)
    st.dataframe(df_info, use_container_width=True, hide_index=True)

# 충전/방전 구간별 분석
st.markdown("---")
st.subheader("🔋 충전/방전 구간별 분석")

# 구간별 탭 생성
tab1, tab2, tab3 = st.tabs(["⚡ 급속충전구간", "🔌 완속충전구간", "📉 방전구간"])

with tab1:
    st.subheader("⚡ 급속충전구간 분석")
    # 급속충전 관련 그래프와 분석 결과가 들어갈 공간
    st.write("급속충전 구간 데이터 분석 결과")
    
    with st.expander("📸 급속충전 그래프 보기", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image('/Users/moon/ev_streamlit/images/fastcharge1.png', caption='완속충전1', use_column_width=True)

        with col2:
            st.image('/Users/moon/ev_streamlit/images/fastcharge2.png', caption='완속충전2', use_column_width=True)

        with col3:
            st.image('/Users/moon/ev_streamlit/images/fastcharge3.png', caption='완속충전3', use_column_width=True)

    st.markdown("""
    - **SOC 평균**은 빠르게 증가 → 빠른 충전 진행
    - **Current 평균**은 초기에 높고, 점차 감소하거나 급락 → 충전 프로파일 반영
    - **Temperature 평균**은 완만하게 상승하거나 안정적 → 과열 없이 안정적 충전
    - **Cell 평균**은 낮은 수준에서 일정하거나 완만한 상승 → 셀 전압 안정적
    - **정규화 그래프**에서도 유사한 상승/하강 패턴 반복
    - **히트맵 상관관계**는 soc와 current/temperature 간의 중간 양의 상관성
    """)
    
with tab2:
    st.subheader("🔌 완속충전구간 분석")
    # 완속충전 관련 그래프와 분석 결과가 들어갈 공간
    st.write("완속충전 구간 데이터 분석 결과")
    with st.expander("📸 완속충전 그래프 보기", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image('/Users/moon/ev_streamlit/images/slowcharge1.png', caption='완속충전1', use_column_width=True)

        with col2:
            st.image('/Users/moon/ev_streamlit/images/slowcharge2.png', caption='완속충전2', use_column_width=True)

        with col3:
            st.image('/Users/moon/ev_streamlit/images/slowcharge3.png', caption='완속충전3', use_column_width=True)

    st.write("""
    1. SOC(평균 충전 상태)의 완만한 증가

    2. Current(전류)의 매우 낮은 값 유지
    current_평균은 대부분 0 또는 0에 가까운 낮은 값을 유지하거나 미세한 변동만 있습니다.(완속 충전에서 저전류로 장시간 충전하는 특성)
    
    3. Temperature(온도)의 거의 일정한 유지(고전류로 인한 발열이 없기 때문에 온도 변화가 거의 없음)

    4. Cell 평균의 변화 거의 없음(일정하거나 미세한 변동을 보이며 큰 패턴 변화가 없음, 안정적인 충전 환경에서 셀 전압 변화가 크지 않음)


    5. SOC만이 상대적으로 가장 일정한 상승 패턴을 유지합니다.

    6. 상관관계 히트맵에서 유사한 색상 구조
    모든 히트맵에서 soc, temperature, current, cell 간의 상관관계는 약하거나 중립적인 수준
    """)
with tab3:
    st.subheader("📉 방전구간 분석")
    # 방전 관련 그래프와 분석 결과가 들어갈 공간
    st.write("방전 구간 데이터 분석 결과")
import streamlit as st
import pandas as pd
import numpy as np
import os

# --- 기본 전처리 규칙 ---
def get_default_rules():
    return [
        {'pattern': 'soc', 'min': 0.0, 'max': 99.0},
        {'pattern': 'soh', 'min': 0.0, 'max': 99.0},
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

st.title("결측치 데이터 전처리")


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

st.subheader("전처리 파일 경로")
folder = st.text_input("CSV 폴더 경로", value="./data")

def list_csv_files_local(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

def get_violation_counts_custom(df, rules):
    result = {}
    for rule in rules:
        pattern = rule['pattern']
        min_v = rule['min']
        max_v = rule['max']
        for col in [c for c in df.columns if pattern in c]:
            result[f"{col} ({min_v}~{max_v} 위반)"] = df[(df[col].notna()) & ((df[col] < min_v) | (df[col] > max_v))].shape[0]
    return result

def custom_preprocessing(df, rules):
    for rule in rules:
        pattern = rule['pattern']
        min_v = rule['min']
        max_v = rule['max']
        for col in [c for c in df.columns if pattern in c]:
            df[col] = df[col].map(lambda v: np.nan if (isinstance(v, (int, float)) and (v < min_v or v > max_v)) else v)
    return df

if st.button("미리보기"):
    csv_files = list_csv_files_local(folder)
    stats = []
    with st.spinner("규칙 위반 row 집계 중..."):
        for f in csv_files:
            df = pd.read_csv(f)
            violations = get_violation_counts_custom(df, st.session_state['rules'])
            violations['file'] = os.path.basename(f)
            stats.append(violations)
    st.write("파일별 전처리 규칙 위반 row 수")
    st.dataframe(pd.DataFrame(stats).fillna(0))

if st.button("전처리 시작"):
    csv_files = list_csv_files_local(folder)
    progress = st.progress(0)
    for i, f in enumerate(csv_files):
        df = pd.read_csv(f)
        df_proc = custom_preprocessing(df, st.session_state['rules'])
        save_path = os.path.join(folder, "preproc_" + os.path.basename(f))
        df_proc.to_csv(save_path, index=False)
        progress.progress((i+1)/len(csv_files))
    st.success("전처리 완료!")
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import math
import warnings
import random
import time
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io

# ------------------------------------------------------------------------------
# [설정] 페이지 기본 설정
# ------------------------------------------------------------------------------
st.set_page_config(page_title="FDTS 시뮬레이터", page_icon="📈", layout="wide")

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# [CSS] 스타일 커스텀
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    /* 전체 폰트 및 스타일 조정 */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* 카드 스타일 (HTML 대시보드용) */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9f9f9 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #eee;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
    }
    .card-title { font-size: 12px; color: #7f8c8d; font-weight: bold; text-transform: uppercase; margin-bottom: 5px; }
    .card-value { font-size: 20px; font-weight: 900; color: #2c3e50; margin-bottom: 0; }
    .sub-value { font-size: 11px; color: #95a5a6; margin-top: 3px; }
    
    /* 테이블 스타일 */
    .styled-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .styled-table th { background-color: #f4f4f4; color: #333; font-weight: bold; text-align: center; padding: 8px; }
    .styled-table td { padding: 6px; text-align: center; border-bottom: 1px solid #ddd; }
    .styled-table tr:nth-child(even) { background-color: #fafafa; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# [0] 전략 데이터베이스
# ==============================================================================
STRATEGY_DB = {
    '1. 터보 운전법': {
        'split': 7, 'profit': 85.0, 'loss': 35.0, 'cycle': 9,
        'mode_logic': 'Standard', 'use_mode': True,
        'active_modes': ['Turbo', 'Sports', 'Comfort', 'Eco'],
        'weights': {1: 0.0, 2: 0.3, 3: 0.5, 4: 0.7, 5: 2.3, 6: 2.2, 7: 1.0},
        'rules': {"Turbo": {"Buy": 2.8, "Sell": 2.6}, "Sports": {"Buy": 4.1, "Sell": 3.2}, "Comfort": {"Buy": 5.3, "Sell": 2.1}, "Eco": {"Buy": 6.6, "Sell": 0.4}},
        'sl_matrix': {"Turbo": [6, 7, 8], "Sports": [7, 8, 10], "Comfort": [16, 18, 20], "Eco": [26, 27, 30]}
    },
    '2. 안전 운전법': {
        'split': 7, 'profit': 75.0, 'loss': 40.0, 'cycle': 10,
        'mode_logic': 'Standard', 'use_mode': True,
        'active_modes': ['Turbo', 'Sports', 'Comfort', 'Eco'],
        'weights': {1: 0.0, 2: 0.0, 3: 0.0, 4: 1.1, 5: 2.3, 6: 2.4, 7: 1.3},
        'rules': {"Turbo": {"Buy": 3.5, "Sell": 2.8}, "Sports": {"Buy": 4.5, "Sell": 2.8}, "Comfort": {"Buy": 5.0, "Sell": 2.0}, "Eco": {"Buy": 6.5, "Sell": 0.6}},
        'sl_matrix': {"Turbo": [6, 7, 8], "Sports": [6, 7, 8], "Comfort": [15, 17, 20], "Eco": [25, 28, 30]}
    },
    '3. 풍차 매매법': {
        'split': 10, 'profit': 90.0, 'loss': 25.0, 'cycle': 5,
        'mode_logic': 'Standard', 'use_mode': True,
        'active_modes': ['Turbo', 'Sports', 'Comfort', 'Eco'],
        'weights': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.5, 7: 3.0, 8: 4.0, 9: 2.0, 10: 0.5},
        'rules': {"Turbo": {"Buy": 3.5, "Sell": 0.1}, "Sports": {"Buy": 4.5, "Sell": 0.1}, "Comfort": {"Buy": 5.0, "Sell": 0.1}, "Eco": {"Buy": 6.5, "Sell": 0.1}},
        'sl_matrix': {"Turbo": [10, 15, 20], "Sports": [12, 17, 22], "Comfort": [15, 20, 25], "Eco": [20, 25, 30]}
    },
    '4. 동파법': {
        'split': 7, 'profit': 80.0, 'loss': 30.0, 'cycle': 10,
        'mode_logic': 'Dongpa', 'use_mode': True,
        'active_modes': ['Sports', 'Eco'],
        'weights': {i: 1.0 for i in range(1, 101)},
        'rules': {"Turbo": {"Buy": 0.0, "Sell": 0.0}, "Sports": {"Buy": 5.0, "Sell": 2.5}, "Comfort": {"Buy": 0.0, "Sell": 0.0}, "Eco": {"Buy": 3.0, "Sell": 0.2}},
        'sl_matrix': {"Turbo": [0, 0, 0], "Sports": [7, 7, 7], "Comfort": [0, 0, 0], "Eco": [30, 30, 30]}
    },
    '5. 떨사오팔': {
        'split': 7, 'profit': 80.0, 'loss': 30.0, 'cycle': 10,
        'mode_logic': 'Single_Eco', 'use_mode': False,
        'active_modes': ['Eco'],
        'weights': {i: 1.0 for i in range(1, 101)},
        'rules': {"Comfort": {"Buy": -0.1, "Sell": 0.1}, "Eco": {"Buy": -0.1, "Sell": 0.1}}, # Eco 추가 매핑
        'sl_matrix': {"Comfort": [30, 30, 30], "Eco": [30, 30, 30]}
    },
    '6. 종사종팔3': {
        'split': 7, 'profit': 70.0, 'loss': 0.0, 'cycle': 10,
        'mode_logic': 'Single_Eco', 'use_mode': False,
        'active_modes': ['Eco'],
        'weights': {i: 1.0 for i in range(1, 101)},
        'rules': {"Turbo": {"Buy": 15.0, "Sell": 2.7}, "Sports": {"Buy": 15.0, "Sell": 2.7}, "Comfort": {"Buy": 15.0, "Sell": 2.7}, "Eco": {"Buy": 15.0, "Sell": 2.7}},
        'sl_matrix': {"Turbo": [10, 10, 10], "Sports": [10, 10, 10], "Comfort": [10, 10, 10], "Eco": [10, 10, 10]}
    }
}
STRATEGY_EN_MAP = {'1. 터보 운전법': 'Turbo', '2. 안전 운전법': 'Safety', '3. 풍차 매매법': 'WindWheel', '4. 동파법': 'DSS', '5. 떨사오팔': '0458', '6. 종사종팔3': 'Jong3'}

# --- 세션 상태 초기화 및 연동 ---
if 's_name' not in st.session_state:
    st.session_state.s_name = list(STRATEGY_DB.keys())[0]

def update_defaults_on_strategy_change():
    new_strategy_name = st.session_state.s_name
    config = STRATEGY_DB[new_strategy_name]
    
    st.session_state.split = int(config['split'])
    st.session_state.p_rate = float(config['profit'])
    st.session_state.l_rate = float(config['loss'])
    st.session_state.cycle = int(config['cycle'])
    
    modes = ['Turbo', 'Sports', 'Comfort', 'Eco']
    for mode in modes:
        mode_rules = config['rules'].get(mode, {'Buy': 0.0, 'Sell': 0.0})
        st.session_state[f"param_{mode}_Buy"] = float(mode_rules.get('Buy', 0.0))
        st.session_state[f"param_{mode}_Sell"] = float(mode_rules.get('Sell', 0.0))
        
        sl_list = config['sl_matrix'].get(mode, [0, 0, 0])
        sl_list = sl_list if len(sl_list) >= 3 else [0, 0, 0]
        st.session_state[f"param_{mode}_SL_H"] = int(sl_list[0])
        st.session_state[f"param_{mode}_SL_M"] = int(sl_list[1])
        st.session_state[f"param_{mode}_SL_L"] = int(sl_list[2])
        
    for i in range(1, 11):
        st.session_state[f"weight_{i}"] = float(config['weights'].get(i, 0.0))

# 초기값 세팅
if 'split' not in st.session_state:
    st.session_state.ticker = "SOXL"
    st.session_state.method = '정수매수 (분모=목표가)'
    st.session_state.seed = 40000
    st.session_state.commission = 0.044
    st.session_state.start_d = date(2025, 1, 1)
    st.session_state.end_d = datetime.now().date()
    update_defaults_on_strategy_change()


# ==============================================================================
# [1] UI 레이아웃 구성 (3단 컬럼)
# ==============================================================================
st.title("📊 FDTS Premium Dashboard v25.1")
st.markdown("---")

# 레이아웃 컨테이너
col_left, col_center, col_right = st.columns([1.1, 1.0, 0.9], gap="medium")

with col_left:
    st.subheader("🛠️ 기본 설정")
    with st.container(border=True):
        st.selectbox("📌 매매전략", list(STRATEGY_DB.keys()), key='s_name', on_change=update_defaults_on_strategy_change)
        
        c1, c2 = st.columns(2)
        c1.text_input("📈 종목코드", key='ticker')
        c2.selectbox("⚖️ 매수방식", ['정액매수 (분모=종가)', '정수매수 (분모=목표가)'], key='method')
        
        c3, c4 = st.columns(2)
        c3.number_input("💰 초기자본($)", step=1000, key='seed')
        c4.number_input("🔢 분할수", min_value=1, step=1, key='split')
        
        c5, c6 = st.columns(2)
        c5.number_input("🔄 갱신주기(일)", min_value=1, step=1, key='cycle')
        c6.number_input("💸 수수료율(%)", step=0.001, format="%.3f", key='commission')

        c7, c8 = st.columns(2)
        c7.date_input("📅 시작일", key='start_d')
        c8.date_input("🏁 종료일", key='end_d')
        
        c9, c10 = st.columns(2)
        c9.number_input("🔺 이익복리(%)", step=0.1, key='p_rate')
        c10.number_input("🔻 손실복리(%)", step=0.1, key='l_rate')

with col_center:
    st.subheader("⚙️ 파라미터 튜닝")
    with st.container(border=True):
        modes = ['Turbo', 'Sports', 'Comfort', 'Eco']
        params_labels = ['매수율(%)', '익절율(%)', 'SL(상단)', 'SL(중단)', 'SL(하단)']
        param_keys = ['Buy', 'Sell', 'SL_H', 'SL_M', 'SL_L']
        
        # 헤더
        cols = st.columns([1.5, 1, 1, 1, 1])
        cols[0].markdown("**항목**")
        for i, m in enumerate(modes): cols[i+1].markdown(f"**{m[0]}**") # T, S, C, E

        # 입력 필드
        for r_idx, label in enumerate(params_labels):
            p_key = param_keys[r_idx]
            cols = st.columns([1.5, 1, 1, 1, 1])
            cols[0].caption(label)
            
            for c_idx, mode in enumerate(modes):
                key_id = f"param_{mode}_{p_key}"
                with cols[c_idx+1]:
                    if 'SL' in p_key:
                        st.number_input("SL", key=key_id, min_value=0, step=1, label_visibility="collapsed")
                    else:
                        st.number_input("Rate", key=key_id, step=0.1, format="%.1f", label_visibility="collapsed")

with col_right:
    st.subheader("⚖️ 비중 설정")
    with st.container(border=True):
        split_val = st.session_state.split
        
        # 2열 그리드로 비중 표시
        cols_w = st.columns(2)
        for i in range(1, 11):
            key_id = f"weight_{i}"
            if i <= split_val:
                idx = (i-1) % 2
                cols_w[idx].number_input(f"{i}차", key=key_id, step=0.1)

# 최적화 패널
st.markdown("---")
with st.expander("🎯 Deep Mine (최적화) 설정", expanded=False):
    col_opt1, col_opt2 = st.columns([3, 1])
    with col_opt1:
        opt_target = st.selectbox("최적화 목표", [
            ('1. 각 모드별 매수/매도율 최적화', 'rates'),
            ('2. 각 모드별 손절일 최적화', 'sl'),
            ('3. 분할별 비중 최적화', 'weights'),
            ('4. 전체 동시 최적화', 'all')
        ], format_func=lambda x: x[0])
    with col_opt2:
        opt_iter = st.selectbox("탐색 횟수", [100, 300, 500, 1000, 2000, 5000], index=1)

# 실행 버튼
col_btn1, col_btn2 = st.columns(2)
run_clicked = col_btn1.button("✨ 시뮬레이션 시작", type="primary", use_container_width=True)
opt_clicked = col_btn2.button("⛏️ Deep Mine 실행", type="secondary", use_container_width=True)

# ==============================================================================
# [3] 핵심 로직 (Backend)
# ==============================================================================
@st.cache_data
def fetch_data(ticker, start_date, end_date):
    buffer_date = start_date - timedelta(weeks=60)
    qqq = yf.download("QQQ", start=buffer_date, end=end_date + timedelta(days=1), auto_adjust=False, progress=False)
    target = yf.download(ticker, start=buffer_date, end=end_date + timedelta(days=1), auto_adjust=False, progress=False)
    
    if isinstance(qqq.columns, pd.MultiIndex): qqq.columns = qqq.columns.get_level_values(0)
    if isinstance(target.columns, pd.MultiIndex): target.columns = target.columns.get_level_values(0)
    
    df_full = target.copy()
    df_full['QQQ_Close'] = qqq['Close']
    # QQQ 데이터 길이가 다를 수 있으므로 정렬 및 ffill
    df_full['QQQ_Close'] = df_full['QQQ_Close'].ffill()
    
    return df_full

def calculate_fdts(df_raw, ticker, seed, split, cycle, profit_rate, loss_rate, commission_rate, method, weights, rules, sl_matrix, use_mode, mode_logic, start_date, end_date):
    # RSI 및 모드 계산
    q_weekly = df_raw['QQQ_Close'].resample('W-FRI').last().to_frame()
    delta = q_weekly['QQQ_Close'].diff()
    up = delta.clip(lower=0).rolling(14).mean(); down = (-1 * delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan); q_weekly['wRSI'] = 100 - (100 / (1 + rs))
    q_weekly['RSI_1'] = q_weekly['wRSI'].shift(1); q_weekly['RSI_2'] = q_weekly['wRSI'].shift(2)

    modes_std, modes_dp = [], []; p_std, p_dp = "Comfort", "Eco"
    for _, row in q_weekly.iterrows():
        r1, r2 = row['RSI_1'], row['RSI_2']; m_std = p_std
        if not (pd.isna(r1) or pd.isna(r2)):
            if (r2 < 40) and ((r1 - r2) >= 5) and (r1 <= 55): m_std = "Turbo"
            elif ((r2 > 65 and r1 < r2) or (40 < r2 < 50 and r1 < r2) or (r1 < 50 and r2 > 50)): m_std = "Eco"
            elif ((r2 < 35 and r1 > r2) or (50 < r2 < 60 and r1 > r2) or (r1 > 50 and r2 < 50)): m_std = "Sports"
            elif (40 <= r2 <= 65): m_std = "Comfort"
        modes_std.append(m_std); p_std = m_std

        m_dp = p_dp
        if not (pd.isna(r1) or pd.isna(r2)):
            if (r2 >= 65 and r1 < r2) or (40 <= r2 <= 50 and r1 < r2) or (r2 >= 50 and r1 < 50): m_dp = "Eco"
            elif (r2 <= 50 and r1 > 50) or (50 <= r2 <= 60 and r1 > r2) or (r2 <= 35 and r1 > r2): m_dp = "Sports"
        modes_dp.append(m_dp); p_dp = m_dp

    q_weekly['Mode_Std'] = modes_std; q_weekly['Mode_Dongpa'] = modes_dp

    # Target 데이터 준비
    target = df_raw[['Close']].copy()
    d_delta = target['Close'].diff()
    up2 = d_delta.clip(lower=0).rolling(14).mean(); down2 = (-1 * d_delta.clip(upper=0)).abs().rolling(14).mean()
    rs2 = up2 / down2.replace(0, np.nan); target['dRSI'] = 100 - (100 / (1 + rs2))
    target['Change'] = target['Close'].pct_change() * 100
    
    target['wRSI'] = q_weekly['wRSI'].reindex(target.index, method='bfill')
    target['Mode_Std'] = q_weekly['Mode_Std'].reindex(target.index, method='bfill').fillna("Comfort")
    target['Mode_Dongpa'] = q_weekly['Mode_Dongpa'].reindex(target.index, method='bfill').fillna("Eco")

    if mode_logic == 'Dongpa': target['Mode'] = target['Mode_Dongpa']
    elif mode_logic == 'Single_Eco': target['Mode'] = 'Eco'
    else: target['Mode'] = target['Mode_Std']

    def get_params(row):
        m = row['Mode']; dr = row['dRSI']
        if not use_mode: m = "Comfort"
        rs_local = rules.get(m, rules.get("Comfort"))
        sl_list = sl_matrix.get(m, [15, 17, 20])
        sl = sl_list[1]
        if pd.notnull(dr):
            if dr >= 58: sl = sl_list[0]
            elif dr <= 40: sl = sl_list[2]
        return pd.Series([rs_local.get("Buy", 0.05), rs_local.get("Sell", 0.02), sl])

    target[['Buy_Rate', 'Sell_Rate', 'SL_Days']] = target.apply(get_params, axis=1)
    target['Prev_Close'] = target['Close'].shift(1)
    target['Target_Price'] = target['Prev_Close'] * (1 + target['Buy_Rate'])

    df = target.loc[start_date:end_date].copy()
    if df.empty: return None

    # 시뮬레이션 변수
    df['Split_Count'] = split; df['Real_Split'] = 0; df['Split_Weight'] = 0.0; df['1_Time_Input'] = 0.0
    df['Input_Asset'] = float(seed); df['Update_Amt'] = 0.0
    df['Is_Buy'] = False; df['Actual_Buy_Price'] = 0.0; df['Buy_Vol'] = 0
    df['Sell_Target_Price'] = np.nan; df['TP_Price'] = np.nan; df['TP_Date'] = None; df['SL_Price'] = np.nan; df['SL_Date'] = None
    df['Status'] = ""; df['Daily_PnL'] = 0.0; df['Daily_Sell_Amt'] = 0.0; df['Total_Buy_Amt'] = 0.0; df['Total_Eval_Amt'] = 0.0
    df['Total_Deposit'] = 0.0; df['Total_Asset'] = 0.0; df['Buy_Fee'] = 0.0; df['Sell_Fee'] = 0.0

    current_real_cash = float(seed); current_input_asset = float(seed); period_net_accum = 0.0
    days_counter = 0; portfolio = []; current_split = 0
    trade_win_cnt = 0; trade_loss_cnt = 0; gross_profit = 0.0; gross_loss = 0.0

    def format_short_date(dt): return dt.strftime("%y/%m/%d").replace("/0", "/")

    for i in range(len(df)):
        days_counter += 1; update_amount = 0.0
        if days_counter > cycle:
            update_amount = period_net_accum * profit_rate if period_net_accum > 0 else period_net_accum * loss_rate
            current_input_asset += update_amount; days_counter = 1; period_net_accum = 0.0
        
        df.iloc[i, df.columns.get_loc('Input_Asset')] = current_input_asset
        df.iloc[i, df.columns.get_loc('Update_Amt')] = update_amount
        curr_date = df.index[i].date(); curr_close = float(df['Close'].iloc[i])

        target_split_level = current_split + 1
        weight = weights.get(target_split_level, 0.0)
        if target_split_level > split: weight = 0.0
        df.iloc[i, df.columns.get_loc('Split_Weight')] = weight
        
        one_time_input = (current_input_asset / split) * weight
        if current_real_cash < 0: one_time_input = 0.0
        else: one_time_input = min(one_time_input, current_real_cash)
        df.iloc[i, df.columns.get_loc('1_Time_Input')] = one_time_input

        daily_status = []; new_portfolio = []; daily_pnl_accum = 0.0; daily_sell_accum = 0.0; sell_occurred_today = 0
        daily_buy_fee = 0.0; daily_sell_fee = 0.0

        for item in portfolio:
            if item['type'] == 'HOLD': new_portfolio.append(item); continue
            if item['sell_date'] == curr_date:
                trade_profit = (item['sell_price'] - item['price']) * item['qty']
                sell_amount = item['sell_price'] * item['qty']
                s_fee = sell_amount * (commission_rate / 100); daily_sell_fee += s_fee
                current_real_cash += (sell_amount - s_fee)
                daily_pnl_accum += trade_profit; daily_sell_accum += sell_amount; period_net_accum += trade_profit
                
                if trade_profit > 0: trade_win_cnt += 1; gross_profit += trade_profit
                elif trade_profit < 0: trade_loss_cnt += 1; gross_loss += abs(trade_profit)
                label = "익절" if item['type'] == 'TP' else "손절"
                if label not in daily_status: daily_status.append(label)
                sell_occurred_today += 1
            else: new_portfolio.append(item)
        portfolio = new_portfolio; current_split -= sell_occurred_today; current_split = max(0, current_split)

        prev_close = df['Prev_Close'].iloc[i]
        if pd.notnull(prev_close):
            target_price = float(df['Target_Price'].iloc[i])
            if curr_close <= target_price:
                df.iloc[i, df.columns.get_loc('Is_Buy')] = True; df.iloc[i, df.columns.get_loc('Actual_Buy_Price')] = curr_close
                buy_vol = 0
                if one_time_input > 0:
                    denominator = curr_close if '정액매수' in method else target_price
                    buy_vol = math.floor(one_time_input / denominator)
                    max_buyable = math.floor(current_real_cash / curr_close)
                    buy_vol = min(buy_vol, max_buyable)
                    
                    buy_amt = buy_vol * curr_close
                    b_fee = buy_amt * (commission_rate / 100); daily_buy_fee += b_fee
                    current_real_cash -= (buy_amt + b_fee)

                if target_split_level <= split:
                    df.iloc[i, df.columns.get_loc('Buy_Vol')] = buy_vol
                    if "매수" not in daily_status: daily_status.append("매수")
                    current_split += 1
                    sell_rate = float(df['Sell_Rate'].iloc[i]); sl_days = int(df['SL_Days'].iloc[i])
                    sell_target = curr_close * (1 + sell_rate)
                    df.iloc[i, df.columns.get_loc('Sell_Target_Price')] = sell_target
                    start_idx = i + 1; target_sl_idx = i + sl_days; end_idx = min(target_sl_idx + 1, len(df))
                    sell_date = None; sell_price_res = 0.0; sell_type_res = 'HOLD'
                    if start_idx < len(df):
                        future_window = df.iloc[start_idx : end_idx]; hit_mask = future_window['Close'] >= sell_target
                        if hit_mask.any():
                            sell_idx = hit_mask.idxmax(); hit_row = df.loc[sell_idx]
                            df.iloc[i, df.columns.get_loc('TP_Price')] = float(hit_row['Close']); df.iloc[i, df.columns.get_loc('TP_Date')] = format_short_date(sell_idx)
                            sell_date = sell_idx.date(); sell_price_res = float(hit_row['Close']); sell_type_res = 'TP'
                        elif sl_days > 0 and target_sl_idx < len(df):
                            sell_idx = df.index[target_sl_idx]; last_row = df.loc[sell_idx]
                            df.iloc[i, df.columns.get_loc('SL_Price')] = float(last_row['Close']); df.iloc[i, df.columns.get_loc('SL_Date')] = format_short_date(sell_idx)
                            sell_date = sell_idx.date(); sell_price_res = float(last_row['Close']); sell_type_res = 'SL'
                    portfolio.append({'qty': int(buy_vol), 'price': float(curr_close), 'sell_date': sell_date, 'sell_price': float(sell_price_res), 'type': sell_type_res})

        total_buy_amt = sum([item['qty'] * item['price'] for item in portfolio])
        total_eval_amt = sum([item['qty'] * curr_close for item in portfolio])
        total_asset = current_real_cash + total_eval_amt
        df.iloc[i, df.columns.get_loc('Status')] = ",".join(daily_status)
        df.iloc[i, df.columns.get_loc('Daily_Sell_Amt')] = daily_sell_accum; df.iloc[i, df.columns.get_loc('Daily_PnL')] = daily_pnl_accum
        df.iloc[i, df.columns.get_loc('Total_Buy_Amt')] = total_buy_amt; df.iloc[i, df.columns.get_loc('Total_Eval_Amt')] = total_eval_amt
        df.iloc[i, df.columns.get_loc('Total_Deposit')] = current_real_cash; df.iloc[i, df.columns.get_loc('Total_Asset')] = total_asset
        df.iloc[i, df.columns.get_loc('Real_Split')] = current_split
        df.iloc[i, df.columns.get_loc('Buy_Fee')] = daily_buy_fee; df.iloc[i, df.columns.get_loc('Sell_Fee')] = daily_sell_fee

    df['Accum_Return'] = (df['Total_Asset'] - float(seed)) / float(seed) * 100
    df['Peak_Asset'] = df['Total_Asset'].cummax(); df['DD'] = (df['Total_Asset'] - df['Peak_Asset']) / df['Peak_Asset'] * 100
    final_asset = float(df['Total_Asset'].iloc[-1]); total_return = (final_asset - seed) / seed * 100
    mdd = float(df['DD'].min()); total_fee = df['Buy_Fee'].sum() + df['Sell_Fee'].sum()
    total_trades = trade_win_cnt + trade_loss_cnt
    win_rate = (trade_win_cnt / total_trades * 100) if total_trades > 0 else 0.0

    return {'df': df, 'total_return': total_return, 'final_asset': final_asset, 'mdd': mdd, 'win_rate': win_rate, 'total_fee': total_fee}

# --- 수집용 함수 ---
def get_current_inputs():
    s_name = st.session_state.s_name; base_config = STRATEGY_DB[s_name]
    ticker = st.session_state.ticker; seed = float(st.session_state.seed); split = int(st.session_state.split)
    start_date = st.session_state.start_d; end_date = st.session_state.end_d
    method = st.session_state.method; commission = float(st.session_state.commission)
    profit = float(st.session_state.p_rate)*0.01; loss = float(st.session_state.l_rate)*0.01; cycle = int(st.session_state.cycle)
    
    rules = {}; sl_mat = {}
    for m in modes:
        b = float(st.session_state[f"param_{m}_Buy_Rate"]) * 0.01
        s = float(st.session_state[f"param_{m}_Sell_Rate"]) * 0.01
        rules[m] = {"Buy": b, "Sell": s}
        h = int(st.session_state[f"param_{m}_SL_High"]); md = int(st.session_state[f"param_{m}_SL_Mid"]); l = int(st.session_state[f"param_{m}_SL_Low"])
        sl_mat[m] = [h, md, l]
        
    weights = {i: (float(st.session_state[f"weight_{i}"]) if i <= split else 0.0) for i in range(1, 11)}
    
    return ticker, seed, split, cycle, profit, loss, commission, method, weights, rules, sl_mat, base_config['use_mode'], base_config['mode_logic'], base_config.get('active_modes', modes), start_date, end_date

# --- 포맷팅 함수 ---
def format_params_compact(rules, sl_mat, weights, split_cnt, active_modes):
    abbr = {'Turbo': 'T', 'Sports': 'S', 'Comfort': 'C', 'Eco': 'E'}
    r_list = [f"{abbr.get(m, m[0])}({rules[m]['Buy']*100:.1f}/{rules[m]['Sell']*100:.1f})" for m in active_modes]
    s_list = [f"{abbr.get(m, m[0])}{sl_mat[m]}" for m in active_modes]
    w_list = [weights[i] for i in range(1, split_cnt + 1)]
    return " ".join(r_list), " ".join(s_list), str(w_list).replace(" ", "")

# ==============================================================================
# [4] 실행 핸들러
# ==============================================================================
if run_clicked:
    inputs = list(get_current_inputs())
    calc_args = [inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[11], inputs[12], inputs[14], inputs[15]]
    
    with st.spinner(f"🚀 {inputs[0]} 시뮬레이션 중..."):
        try:
            df_full = fetch_data(inputs[0], inputs[14], inputs[15])
            if df_full.empty: st.error("데이터 없음"); st.stop()
            
            res = calculate_fdts(df_full, *calc_args)
            if not res: st.error("결과 계산 실패"); st.stop()
            
            df = res['df']
            # CAGR 계산
            days = (df.index[-1] - df.index[0]).days; years = days/365.25
            cagr = ((res['final_asset']/inputs[1])**(1/years)-1)*100 if years > 0 else 0.0
            
            # --- 📊 Dashboard ---
            def color_val(val): return "#e74c3c" if val >= 0 else "#2980b9"
            cols = st.columns(6)
            cols[0].markdown(f"**Total Return**\n\n<span style='color:{color_val(res['total_return'])}; font-size:20px; font-weight:bold'>{res['total_return']:+.0f}%</span>", unsafe_allow_html=True)
            cols[1].markdown(f"**Final Asset**\n\n<span style='color:#2c3e50; font-size:20px; font-weight:bold'>${res['final_asset']:,.0f}</span>", unsafe_allow_html=True)
            cols[2].markdown(f"**CAGR**\n\n<span style='color:#8e44ad; font-size:20px; font-weight:bold'>{cagr:.1f}%</span>", unsafe_allow_html=True)
            cols[3].markdown(f"**MDD**\n\n<span style='color:#2980b9; font-size:20px; font-weight:bold'>{res['mdd']:.2f}%</span>", unsafe_allow_html=True)
            cols[4].markdown(f"**Win Rate**\n\n<span style='color:#27ae60; font-size:20px; font-weight:bold'>{res['win_rate']:.1f}%</span>", unsafe_allow_html=True)
            cols[5].markdown(f"**Total Fee**\n\n<span style='color:#e67e22; font-size:20px; font-weight:bold'>${res['total_fee']:,.0f}</span>", unsafe_allow_html=True)

            # --- 🖼️ Chart ---
            fig = plt.figure(figsize=(16, 10)); gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
            ax1 = plt.subplot(gs[0])
            line1 = ax1.plot(df.index, df['Total_Asset'], label='Asset', color='#e74c3c', linewidth=2)
            ax1.fill_between(df.index, df['Total_Asset'], df['Total_Asset'].min(), color='#e74c3c', alpha=0.05)
            ax1_t = ax1.twinx(); line2 = ax1_t.plot(df.index, df['Close'], label='Price', color='#95a5a6', alpha=0.5, linestyle='--')
            tp_df = df[df['Status'].str.contains('익절', na=False)]; sl_df = df[df['Status'].str.contains('손절', na=False)]
            ax1.scatter(tp_df.index, tp_df['Total_Asset'], marker='^', color='#e74c3c', s=80, zorder=5)
            ax1.scatter(sl_df.index, sl_df['Total_Asset'], marker='v', color='#2980b9', s=80, zorder=5)
            ax1.set_title(f"Asset Growth ({inputs[0]})", fontsize=14, fontweight='bold'); ax1.grid(True, linestyle=':', alpha=0.5)
            ax1.legend(line1+line2, [l.get_label() for l in line1+line2], loc='upper left')
            
            ax2 = plt.subplot(gs[1], sharex=ax1)
            ax2.plot(df.index, df['DD'], color='#2980b9'); ax2.fill_between(df.index, df['DD'], 0, color='#2980b9', alpha=0.2)
            ax2.set_title("Drawdown", fontsize=11, fontweight='bold'); ax2.grid(True, linestyle=':', alpha=0.5)
            
            ax3 = plt.subplot(gs[2], sharex=ax1)
            colors = ['#2980b9' if v < 0 else '#e74c3c' for v in df['Daily_PnL']]
            ax3.bar(df.index, df['Daily_PnL'], color=colors, alpha=0.8); ax3.set_title("Daily PnL", fontsize=11, fontweight='bold'); ax3.grid(True, linestyle=':', alpha=0.5)
            
            st.pyplot(fig)

            # --- 📄 Table ---
            st.subheader("📋 상세 거래 내역")
            cols = ['Close', 'Change', 'wRSI', 'dRSI', 'Mode', 'Buy_Rate', 'Sell_Rate', 'SL_Days', 'Real_Split', 'Input_Asset', 'Split_Count', 'Split_Weight', '1_Time_Input', 'Update_Amt', 'Target_Price', 'Actual_Buy_Price', 'Buy_Vol', 'Sell_Target_Price', 'TP_Price', 'TP_Date', 'SL_Price', 'SL_Date', 'Status', 'Daily_Sell_Amt', 'Daily_PnL', 'Total_Buy_Amt', 'Total_Eval_Amt', 'Total_Deposit', 'Total_Asset', 'Accum_Return', 'DD', 'Buy_Fee', 'Sell_Fee']
            df_disp = df[cols].copy()
            col_map = {'Close': '종가', 'Change': '등락', 'Mode': '모드', 'Buy_Rate': '매수율', 'Sell_Rate': '익절율', 'SL_Days': '손절(일)', 'Real_Split': '분할', 'Input_Asset': '투입자산', 'Split_Count': '설정분할', 'Split_Weight': '비중', '1_Time_Input': '1회투입', 'Update_Amt': '갱신금', 'Target_Price': '매수목표', 'Actual_Buy_Price': '실매수', 'Buy_Vol': '매수량', 'Sell_Target_Price': '매도목표', 'TP_Price': '익절가', 'TP_Date': '익절일', 'SL_Price': '손절가', 'SL_Date': '손절일', 'Status': '상태', 'Daily_Sell_Amt': '매도액', 'Daily_PnL': '손익', 'Total_Buy_Amt': '매수총액', 'Total_Eval_Amt': '평가총액', 'Total_Deposit': '예수금', 'Total_Asset': '자산', 'Accum_Return': '수익률', 'DD': 'DD', 'Buy_Fee': '매수수수료', 'Sell_Fee': '매도수수료'}
            df_disp.rename(columns=col_map, inplace=True)
            df_disp.index = df_disp.index.strftime('%Y-%m-%d')
            
            def highlight(s):
                if s.Status == '상태': return [''] * len(s)
                if '익절' in str(s['상태']): return ['background-color: rgba(231, 76, 60, 0.1); color: #c0392b'] * len(s)
                if '손절' in str(s['상태']): return ['background-color: rgba(41, 128, 185, 0.1); color: #2980b9'] * len(s)
                return [''] * len(s)

            st.dataframe(df_disp.style.apply(highlight, axis=1).format("{:.2f}", subset=['종가', '등락', 'wRSI', 'dRSI', '익절가', '손절가', '매도목표', '매수목표', '실매수', '수익률', 'DD']).format("{:,.0f}", subset=['투입자산', '1회투입', '갱신금', '매수량', '매도액', '손익', '매수총액', '평가총액', '예수금', '자산']), use_container_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ==============================================================================
# [5] 최적화 핸들러
# ==============================================================================
if opt_clicked:
    inputs = list(get_current_inputs())
    ticker_opt = inputs[0]
    
    st.info("⛏️ Deep Mine 최적화 시작... (잠시만 기다려주세요)")
    
    try:
        # 데이터 로드 (한 번만)
        df_full = fetch_data(ticker_opt, inputs[14], inputs[15])
        
        target_opt = opt_target[1] # rates, sl, weights, all
        active_modes = inputs[13]
        base_rules = inputs[9]
        base_sl = inputs[10]
        base_weights = inputs[8]
        split_cnt = inputs[2]
        
        N_ITER = int(opt_iter)
        calc_args_base = [inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[11], inputs[12], inputs[14], inputs[15]]
        
        results = []
        best_cagr = -999.0
        
        progress_text = st.empty()
        bar = st.progress(0)
        
        for i in range(N_ITER):
            curr_rules = {k: v.copy() for k, v in base_rules.items()}
            curr_sl = {k: v.copy() for k, v in base_sl.items()}
            curr_weights = base_weights.copy()
            
            # Randomize based on target
            if target_opt in ['rates', 'all']:
                for m in active_modes:
                    curr_rules[m]['Buy'] = round(curr_rules[m]['Buy']*100 + random.uniform(-1.0, 1.0), 1) * 0.01
                    curr_rules[m]['Sell'] = max(0.001, round(curr_rules[m]['Sell']*100 + random.uniform(-0.5, 0.5), 1) * 0.01)
            
            if target_opt in ['sl', 'all']:
                for m in active_modes:
                    # 단순화: 기존 값에서 +/- random
                    curr_sl[m] = [max(1, x + random.randint(-3, 3)) for x in curr_sl[m]]
                    curr_sl[m].sort()
            
            if target_opt in ['weights', 'all']:
                # 기존 비중에서 약간 변형
                w_list = [curr_weights[x] for x in range(1, split_cnt+1)]
                new_w = [max(0, x + random.uniform(-0.5, 0.5)) for x in w_list]
                # Normalize? 아니면 그냥 둠? -> 합계 일정하게 조정
                if sum(new_w) == 0: new_w = [1]*split_cnt
                factor = sum(w_list) / sum(new_w) if sum(w_list) > 0 else 1
                new_w = [round(x * factor, 1) for x in new_w]
                for idx, val in enumerate(new_w): curr_weights[idx+1] = val

            # Run Calc
            args = calc_args_base.copy()
            args[8] = curr_weights
            args[9] = curr_rules
            args[10] = curr_sl
            
            res = calculate_fdts(df_full, *args)
            if not res: continue
            
            # Calc CAGR
            days = (res['df'].index[-1] - res['df'].index[0]).days
            years = days / 365.25
            cagr = ((res['final_asset'] / inputs[1]) ** (1 / years) - 1) * 100 if years > 0 and res['final_asset'] > 0 else 0
            
            if cagr > best_cagr: best_cagr = cagr
            
            results.append({
                'CAGR': cagr, 'MDD': res['mdd'], 'WinRate': res['win_rate'],
                'Final': res['final_asset'],
                'Settings': format_params_compact(curr_rules, curr_sl, curr_weights, split_cnt, active_modes)
            })
            
            if i % 10 == 0:
                bar.progress((i + 1) / N_ITER)
                progress_text.text(f"Simulation {i+1}/{N_ITER} | Best CAGR: {best_cagr:.2f}%")
        
        bar.progress(100)
        st.success("✅ 최적화 완료!")
        
        # Show Top 10
        res_df = pd.DataFrame(results).sort_values(by='CAGR', ascending=False).head(10)
        st.write("🏆 Top 10 Results (by CAGR)")
        st.dataframe(res_df.style.format({'CAGR': "{:.2f}%", 'MDD': "{:.2f}%", 'WinRate': "{:.1f}%", 'Final': "${:,.0f}"}), use_container_width=True)
        
    except Exception as e:
        st.error(f"Optimization Error: {str(e)}")

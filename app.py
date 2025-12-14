import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import math
import warnings
import random
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io

# ------------------------------------------------------------------------------
# [1] 페이지 설정
# ------------------------------------------------------------------------------
st.set_page_config(page_title="FDTS 시뮬레이터", page_icon="📈", layout="wide")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    .stButton>button { width: 100%; font-weight: bold; border-radius: 8px; height: 3em; }
    .metric-card {
        background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px;
        padding: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# [2] 전략 데이터베이스
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
        'rules': {"Comfort": {"Buy": -0.1, "Sell": 0.1}, "Eco": {"Buy": -0.1, "Sell": 0.1}},
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

# ==============================================================================
# [3] 초기화 및 상태 관리
# ==============================================================================
run_clicked = False 

if 's_name' not in st.session_state:
    st.session_state.s_name = list(STRATEGY_DB.keys())[0]

def update_defaults():
    s_name = st.session_state.s_name
    config = STRATEGY_DB[s_name]
    st.session_state.split = int(config['split'])
    st.session_state.p_rate = float(config['profit'])
    st.session_state.l_rate = float(config['loss'])
    st.session_state.cycle = int(config['cycle'])
    
    modes = ['Turbo', 'Sports', 'Comfort', 'Eco']
    for mode in modes:
        mode_rules = config['rules'].get(mode, {'Buy': 0.0, 'Sell': 0.0})
        st.session_state[f"param_{mode}_Buy_Rate"] = float(mode_rules.get('Buy', 0.0))
        st.session_state[f"param_{mode}_Sell_Rate"] = float(mode_rules.get('Sell', 0.0))
        sl_list = config['sl_matrix'].get(mode, [0, 0, 0])
        sl_list = sl_list if len(sl_list) >= 3 else [0, 0, 0]
        st.session_state[f"param_{mode}_SL_High"] = int(sl_list[0])
        st.session_state[f"param_{mode}_SL_Mid"] = int(sl_list[1])
        st.session_state[f"param_{mode}_SL_Low"] = int(sl_list[2])
    
    for i in range(1, 11):
        st.session_state[f"weight_{i}"] = float(config['weights'].get(i, 0.0))

if 'split' not in st.session_state:
    st.session_state.ticker = "SOXL"
    st.session_state.method = '정수매수 (분모=목표가)'
    st.session_state.seed = 40000
    st.session_state.commission = 0.044
    st.session_state.start_d = date(2025, 1, 1)
    st.session_state.end_d = datetime.now().date()
    update_defaults()

# ==============================================================================
# [4] 사이드바 설정 패널
# ==============================================================================
with st.sidebar:
    st.title("🎛️ FDTS 설정")
    st.selectbox("📌 매매전략", list(STRATEGY_DB.keys()), key='s_name', on_change=update_defaults)
    
    col1, col2 = st.columns(2)
    col1.text_input("📈 종목", key='ticker')
    col2.selectbox("⚖️ 방식", ['정액매수 (분모=종가)', '정수매수 (분모=목표가)'], key='method')
    
    col3, col4 = st.columns(2)
    col3.number_input("💰 자본($)", step=1000, key='seed')
    col4.number_input("🔢 분할수", min_value=1, step=1, key='split')
    
    col5, col6 = st.columns(2)
    col5.number_input("🔄 주기(일)", min_value=1, key='cycle')
    col6.number_input("💸 수수료(%)", step=0.001, format="%.3f", key='commission')
    
    col7, col8 = st.columns(2)
    col7.number_input("🔺 이익(%)", step=0.1, key='p_rate')
    col8.number_input("🔻 손실(%)", step=0.1, key='l_rate')
    
    st.markdown("---")
    st.caption("📅 시뮬레이션 기간")
    c_start, c_end = st.columns(2)
    c_start.date_input("시작", key='start_d')
    c_end.date_input("종료", key='end_d')
    
    st.markdown("---")
    with st.expander("⚙️ 고급 파라미터 튜닝", expanded=False):
        modes = ['Turbo', 'Sports', 'Comfort', 'Eco']
        params_labels = ['매수율', '익절율', 'SL(상)', 'SL(중)', 'SL(하)']
        param_keys = ['Buy_Rate', 'Sell_Rate', 'SL_High', 'SL_Mid', 'SL_Low']
        cols = st.columns([1.5, 1, 1, 1, 1])
        cols[0].markdown("**항목**")
        for i, m in enumerate(modes): cols[i+1].markdown(f"**{m[0]}**")
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
    
    with st.expander("⚖️ 분할별 비중 설정", expanded=False):
        cols_w = st.columns(2)
        for i in range(1, 11):
            key_id = f"weight_{i}"
            cols_w[(i-1)%2].number_input(f"{i}차", key=key_id, step=0.1)

    st.markdown("---")
    if st.button("🚀 실행", type="primary", use_container_width=True):
        st.session_state.run_sim = True

# ==============================================================================
# [5] 메인 로직
# ==============================================================================
st.title("📊 FDTS Trading Simulator")

@st.cache_data
def get_data(ticker, start_date, end_date):
    buffer_date = start_date - timedelta(weeks=60)
    qqq = yf.download("QQQ", start=buffer_date, end=end_date + timedelta(days=1), auto_adjust=False, progress=False)
    target = yf.download(ticker, start=buffer_date, end=end_date + timedelta(days=1), auto_adjust=False, progress=False)
    if isinstance(qqq.columns, pd.MultiIndex): qqq.columns = qqq.columns.get_level_values(0)
    if isinstance(target.columns, pd.MultiIndex): target.columns = target.columns.get_level_values(0)
    return qqq, target

def run_simulation():
    s_name = st.session_state.s_name
    base_config = STRATEGY_DB[s_name]
    ticker = st.session_state.ticker
    seed = float(st.session_state.seed)
    split = int(st.session_state.split)
    start_date = st.session_state.start_d
    end_date = st.session_state.end_d
    method = st.session_state.method
    profit_rate = float(st.session_state.p_rate) * 0.01
    loss_rate = float(st.session_state.l_rate) * 0.01
    commission = float(st.session_state.commission)
    cycle = int(st.session_state.cycle)

    custom_rules = {}; custom_sl_matrix = {}
    modes = ['Turbo', 'Sports', 'Comfort', 'Eco']
    for m in modes:
        b = float(st.session_state[f"param_{m}_Buy_Rate"]) * 0.01
        s = float(st.session_state[f"param_{m}_Sell_Rate"]) * 0.01
        custom_rules[m] = {"Buy": b, "Sell": s}
        h = int(st.session_state[f"param_{m}_SL_High"])
        md = int(st.session_state[f"param_{m}_SL_Mid"])
        l = int(st.session_state[f"param_{m}_SL_Low"])
        custom_sl_matrix[m] = [h, md, l]
    
    custom_weights = {i: (float(st.session_state[f"weight_{i}"]) if i <= split else 0.0) for i in range(1, 11)}

    with st.spinner(f"{ticker} 데이터 분석 중..."):
        try:
            qqq, target = get_data(ticker, start_date, end_date)
            if qqq.empty or target.empty:
                st.error("데이터를 불러오지 못했습니다.")
                return

            q_weekly = qqq['Close'].resample('W-FRI').last().to_frame()
            delta = q_weekly['Close'].diff()
            up = delta.clip(lower=0).rolling(14).mean(); down = (-1 * delta.clip(upper=0)).abs().rolling(14).mean()
            rs = up / down.replace(0, np.nan); q_weekly['wRSI'] = 100 - (100 / (1 + rs))
            q_weekly['RSI_1'] = q_weekly['wRSI'].shift(1); q_weekly['RSI_2'] = q_weekly['wRSI'].shift(2)

            modes_std = []; modes_dp = []; p_std = "Comfort"; p_dp = "Eco"
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

            target['Change'] = target['Close'].pct_change() * 100
            d_delta = target['Close'].diff()
            up2 = d_delta.clip(lower=0).rolling(14).mean(); down2 = (-1 * d_delta.clip(upper=0)).abs().rolling(14).mean()
            rs2 = up2 / down2.replace(0, np.nan); target['dRSI'] = 100 - (100 / (1 + rs2))

            target['wRSI'] = q_weekly['wRSI'].reindex(target.index, method='bfill')
            target['Mode_Std'] = q_weekly['Mode_Std'].reindex(target.index, method='bfill').fillna("Comfort")
            target['Mode_Dongpa'] = q_weekly['Mode_Dongpa'].reindex(target.index, method='bfill').fillna("Eco")
            target['Mode'] = target['Mode_Dongpa'] if base_config['mode_logic'] == 'Dongpa' else target['Mode_Std']

            def get_params(row):
                m = row['Mode']; dr = row['dRSI']
                if not base_config['use_mode']: m = "Comfort"
                rs_local = custom_rules.get(m, custom_rules.get("Comfort"))
                sl_list = custom_sl_matrix.get(m, [15, 17, 20]); sl = sl_list[1]
                if pd.notnull(dr):
                    if dr >= 58: sl = sl_list[0]
                    elif dr <= 40: sl = sl_list[2]
                return pd.Series([rs_local.get("Buy", 0.05), rs_local.get("Sell", 0.02), sl])

            target[['Buy_Rate', 'Sell_Rate', 'SL_Days']] = target.apply(get_params, axis=1)
            target['Prev_Close'] = target['Close'].shift(1)
            target['Target_Price'] = target['Prev_Close'] * (1 + target['Buy_Rate'])

            df = target.loc[start_date:end_date].copy()
            if df.empty: st.error("해당 기간의 데이터가 없습니다."); return

            df['Split_Count'] = split; df['Real_Split'] = 0; df['Split_Weight'] = 0.0; df['1_Time_Input'] = 0.0
            df['Input_Asset'] = float(seed); df['Update_Amt'] = 0.0
            df['Is_Buy'] = False; df['Actual_Buy_Price'] = 0.0; df['Buy_Vol'] = 0
            df['Sell_Target_Price'] = np.nan; df['TP_Price'] = np.nan; df['TP_Date'] = None
            df['SL_Price'] = np.nan; df['SL_Date'] = None; df['Status'] = ""
            df['Daily_PnL'] = 0.0; df['Daily_Sell_Amt'] = 0.0; df['Total_Buy_Amt'] = 0.0
            df['Total_Eval_Amt'] = 0.0; df['Total_Deposit'] = 0.0; df['Total_Asset'] = 0.0
            df['Buy_Fee'] = 0.0; df['Sell_Fee'] = 0.0
            df['Accum_Return'] = 0.0; df['DD'] = 0.0

            current_real_cash = float(seed); current_input_asset = float(seed); period_net_accum = 0.0
            days_counter = 0; portfolio = []; current_split = 0
            trade_win_cnt = 0; trade_loss_cnt = 0; gross_profit = 0.0; gross_loss = 0.0

            def fmt_date(d): return d.strftime("%y/%m/%d").replace("/0", "/")

            for i in range(len(df)):
                days_counter += 1; update_amount = 0.0
                if days_counter > cycle:
                    update_amount = period_net_accum * profit_rate if period_net_accum > 0 else period_net_accum * loss_rate
                    current_input_asset += update_amount; days_counter = 1; period_net_accum = 0.0
                
                df.iloc[i, df.columns.get_loc('Input_Asset')] = current_input_asset
                df.iloc[i, df.columns.get_loc('Update_Amt')] = update_amount
                curr_date = df.index[i].date(); curr_close = float(df['Close'].iloc[i])

                target_split_level = current_split + 1
                weight = custom_weights.get(target_split_level, 0.0)
                if target_split_level > split: weight = 0.0
                df.iloc[i, df.columns.get_loc('Split_Weight')] = weight
                
                one_time_input = (current_input_asset / split) * weight
                one_time_input = 0.0 if current_real_cash < 0 else min(one_time_input, current_real_cash)
                df.iloc[i, df.columns.get_loc('1_Time_Input')] = one_time_input

                daily_status = []; new_portfolio = []; daily_pnl_accum = 0.0; daily_sell_accum = 0.0; sell_occurred_today = 0
                daily_buy_fee = 0.0; daily_sell_fee = 0.0

                for item in portfolio:
                    if item['type'] == 'HOLD': new_portfolio.append(item); continue
                    if item['sell_date'] == curr_date:
                        trade_profit = (item['sell_price'] - item['price']) * item['qty']
                        sell_amount = item['sell_price'] * item['qty']
                        s_fee = sell_amount * (commission / 100); daily_sell_fee += s_fee
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
                            
                            buy_amt = buy_vol * curr_close; b_fee = buy_amt * (commission / 100); daily_buy_fee += b_fee
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
                                    df.iloc[i, df.columns.get_loc('TP_Price')] = float(hit_row['Close']); df.iloc[i, df.columns.get_loc('TP_Date')] = fmt_date(sell_idx)
                                    sell_date = sell_idx.date(); sell_price_res = float(hit_row['Close']); sell_type_res = 'TP'
                                elif sl_days > 0 and target_sl_idx < len(df):
                                    sell_idx = df.index[target_sl_idx]; last_row = df.loc[sell_idx]
                                    df.iloc[i, df.columns.get_loc('SL_Price')] = float(last_row['Close']); df.iloc[i, df.columns.get_loc('SL_Date')] = fmt_date(sell_idx)
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
            df['Peak_Asset'] = df['Total_Asset'].cummax()
            df['DD'] = (df['Total_Asset'] - df['Peak_Asset']) / df['Peak_Asset'] * 100

            final_asset = float(df['Total_Asset'].iloc[-1])
            total_return = (final_asset - seed) / seed * 100
            mdd = float(df['DD'].min())
            days = (df.index[-1] - df.index[0]).days; years = days/365.25
            cagr = ((final_asset / seed) ** (1 / years) - 1) * 100 if (years > 0 and final_asset > 0) else 0.0
            
            total_trades = trade_win_cnt + trade_loss_cnt
            win_rate = (trade_win_cnt / total_trades * 100) if total_trades > 0 else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.99 if gross_profit > 0 else 0.0)
            total_fee = df['Buy_Fee'].sum() + df['Sell_Fee'].sum()

            cols = st.columns(6)
            def color_val(val): return "green" if val >= 0 else "red"
            cols[0].metric("Total Return", f"{total_return:+.1f}%")
            cols[1].metric("Final Asset", f"${final_asset:,.0f}")
            cols[2].metric("CAGR", f"{cagr:.1f}%")
            cols[3].metric("MDD", f"{mdd:.2f}%")
            cols[4].metric("Win Rate", f"{win_rate:.1f}%")
            cols[5].metric("Total Fee", f"${total_fee:,.0f}")

            fig = plt.figure(figsize=(16, 10)); gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
            ax1 = plt.subplot(gs[0])
            line1 = ax1.plot(df.index, df['Total_Asset'], label='Asset', color='#e74c3c', linewidth=2)
            ax1.fill_between(df.index, df['Total_Asset'], df['Total_Asset'].min(), color='#e74c3c', alpha=0.05)
            ax1_t = ax1.twinx(); line2 = ax1_t.plot(df.index, df['Close'], label='Price', color='#95a5a6', alpha=0.5, linestyle='--')
            tp = df[df['Status'].str.contains('익절', na=False)]; sl = df[df['Status'].str.contains('손절', na=False)]
            ax1.scatter(tp.index, tp['Total_Asset'], marker='^', color='#e74c3c', s=80, zorder=5)
            ax1.scatter(sl.index, sl['Total_Asset'], marker='v', color='#2980b9', s=80, zorder=5)
            ax1.legend(line1+line2, [l.get_label() for l in line1+line2], loc='upper left'); ax1.grid(True, linestyle=':', alpha=0.5)
            ax1.set_title("Asset Growth & Trade Points", fontsize=14, fontweight='bold')
            
            ax2 = plt.subplot(gs[1], sharex=ax1)
            ax2.plot(df.index, df['DD'], color='#2980b9'); ax2.fill_between(df.index, df['DD'], 0, color='#2980b9', alpha=0.2); ax2.set_title("Drawdown", fontsize=11, fontweight='bold'); ax2.grid(True, linestyle=':', alpha=0.5)
            
            ax3 = plt.subplot(gs[2], sharex=ax1)
            colors = ['#2980b9' if v < 0 else '#e74c3c' for v in df['Daily_PnL']]
            ax3.bar(df.index, df['Daily_PnL'], color=colors, alpha=0.8); ax3.set_title("Daily PnL", fontsize=11, fontweight='bold'); ax3.grid(True, linestyle=':', alpha=0.5)
            
            st.pyplot(fig)

            with st.expander("📋 상세 거래 내역 (클릭하여 펼치기)", expanded=False):
                # 🌟 [KEY ERROR FIX] 숫자로 변환 가능한 컬럼만 subset 지정
                # Status는 문자열이므로 subset에서 제외
                target_cols = ['Close', 'Change', 'wRSI', 'dRSI', 'Buy_Rate', 'Sell_Rate', 'SL_Days',
                               'Real_Split', 'Input_Asset', 'Split_Count', 'Split_Weight', '1_Time_Input', 'Update_Amt', 
                               'Target_Price', 'Actual_Buy_Price', 'Buy_Vol', 'Sell_Target_Price', 'TP_Price', 'SL_Price', 
                               'Status', 'Daily_Sell_Amt', 'Daily_PnL', 'Total_Buy_Amt', 'Total_Eval_Amt', 'Total_Deposit', 'Total_Asset', 'Accum_Return', 'DD', 'Buy_Fee', 'Sell_Fee']
                
                # 실제 존재하는 컬럼만 선택
                available_cols = [c for c in target_cols if c in df.columns]
                
                # 표시용 데이터프레임 생성
                df_disp = df[available_cols].copy()
                
                # 날짜 포맷팅
                df_disp.index = df_disp.index.strftime('%Y-%m-%d')
                
                # 하이라이트 함수 (글자색 변경)
                def highlight(s):
                    # 전체 행에 대해 기본값(빈 문자열) 리스트 생성
                    styles = [''] * len(s)
                    
                    # Status 컬럼이 있는지 확인하고 조건 검사
                    if 'Status' in s.index:
                        status_val = str(s['Status'])
                        if '익절' in status_val:
                            return ['color: #e74c3c; font-weight: bold'] * len(s)
                        if '손절' in status_val:
                            return ['color: #2980b9; font-weight: bold'] * len(s)
                        if '매수' in status_val:
                            return ['color: green; font-weight: bold'] * len(s)
                    
                    return styles

                # 숫자형 컬럼 찾기 (포맷팅용)
                numeric_cols = df_disp.select_dtypes(include=['float64', 'int64']).columns

                # 스타일 및 포맷 적용 후 출력
                st.dataframe(
                    df_disp.style
                    .apply(highlight, axis=1)
                    .format("{:.2f}", subset=numeric_cols), # 숫자 컬럼만 .2f 포맷 적용
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ 분석 중 오류 발생: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

if st.session_state.get('run_sim'):
    run_simulation()

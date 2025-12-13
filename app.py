import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import math
import warnings
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io # SVG 출력을 위한 라이브러리

# ------------------------------------------------------------------------------
# [웹 설정] 페이지 기본 설정
# ------------------------------------------------------------------------------
st.set_page_config(page_title="FDTS 시뮬레이터", page_icon="📈", layout="wide")

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ==============================================================================
# [0] 전략 데이터베이스 (로직 동일 유지)
# ==============================================================================
# 주의: 웹 코드로 변환 시, UI에서 입력받는 값은 Streamlit 위젯을 통해 직접 받아야 합니다.
# STRATEGY_DB는 설정값의 기본값 및 로직을 제공합니다.
STRATEGY_DB = {
    '1. 터보 운전법': {
        'split': 7, 'profit': 85.0, 'loss': 35.0, 'cycle': 9,
        'mode_logic': 'Standard', 'use_mode': True,
        'weights': {1: 0.0, 2: 0.3, 3: 0.5, 4: 0.7, 5: 2.3, 6: 2.2, 7: 1.0},
        'rules': {"Turbo": {"Buy": 2.8, "Sell": 2.6}, "Sports": {"Buy": 4.1, "Sell": 3.2}, "Comfort": {"Buy": 5.3, "Sell": 2.1}, "Eco": {"Buy": 6.6, "Sell": 0.4}},
        'sl_matrix': {"Turbo": [6, 7, 8], "Sports": [7, 8, 10], "Comfort": [16, 18, 20], "Eco": [26, 27, 30]}
    },
    '2. 안전 운전법': {
        'split': 7, 'profit': 75.0, 'loss': 40.0, 'cycle': 10,
        'mode_logic': 'Standard', 'use_mode': True,
        'weights': {1: 0.0, 2: 0.0, 3: 0.0, 4: 1.1, 5: 2.3, 6: 2.4, 7: 1.3},
        'rules': {"Turbo": {"Buy": 3.5, "Sell": 2.8}, "Sports": {"Buy": 4.5, "Sell": 2.8}, "Comfort": {"Buy": 5.0, "Sell": 2.0}, "Eco": {"Buy": 6.5, "Sell": 0.6}},
        'sl_matrix': {"Turbo": [6, 7, 8], "Sports": [6, 7, 8], "Comfort": [15, 17, 20], "Eco": [25, 28, 30]}
    },
    '3. 풍차 매매법': {
        'split': 10, 'profit': 90.0, 'loss': 25.0, 'cycle': 5,
        'mode_logic': 'Standard', 'use_mode': True,
        'weights': {i: 1.0 for i in range(1, 11)},
        'rules': {"Turbo": {"Buy": 3.5, "Sell": 0.1}, "Sports": {"Buy": 4.5, "Sell": 0.1}, "Comfort": {"Buy": 5.0, "Sell": 0.1}, "Eco": {"Buy": 6.5, "Sell": 0.1}},
        'sl_matrix': {"Turbo": [10, 15, 20], "Sports": [12, 17, 22], "Comfort": [15, 20, 25], "Eco": [20, 25, 30]}
    },
    '4. 동파법': {
        'split': 7, 'profit': 80.0, 'loss': 30.0, 'cycle': 10,
        'mode_logic': 'Dongpa', 'use_mode': True,
        'weights': {i: 1.0 for i in range(1, 101)},
        'rules': {"Turbo": {"Buy": 0.0, "Sell": 0.0}, "Sports": {"Buy": 5.0, "Sell": 2.5}, "Comfort": {"Buy": 0.0, "Sell": 0.0}, "Eco": {"Buy": 3.0, "Sell": 0.2}},
        'sl_matrix': {"Turbo": [0, 0, 0], "Sports": [7, 7, 7], "Comfort": [0, 0, 0], "Eco": [30, 30, 30]}
    },
    '5. 떨사오팔': {
        'split': 7, 'profit': 80.0, 'loss': 30.0, 'cycle': 10,
        'mode_logic': 'Standard', 'use_mode': False,
        'weights': {i: 1.0 for i in range(1, 101)},
        'rules': {"Comfort": {"Buy": -0.1, "Sell": 0.1}},
        'sl_matrix': {"Comfort": [30, 30, 30]}
    },
    '6. 종사종팔3': {
        'split': 7, 'profit': 70.0, 'loss': 0.0, 'cycle': 10,
        'mode_logic': 'Standard', 'use_mode': False,
        'weights': {i: 1.0 for i in range(1, 101)},
        'rules': {"Turbo": {"Buy": 15.0, "Sell": 2.7}, "Sports": {"Buy": 15.0, "Sell": 2.7}, "Comfort": {"Buy": 15.0, "Sell": 2.7}, "Eco": {"Buy": 15.0, "Sell": 2.7}},
        'sl_matrix': {"Turbo": [10, 10, 10], "Sports": [10, 10, 10], "Comfort": [10, 10, 10], "Eco": [10, 10, 10]}
    }
}
STRATEGY_EN_MAP = {'1. 터보 운전법': 'Turbo Driving', '2. 안전 운전법': 'Safety Driving', '3. 풍차 매매법': 'Wind Wheel', '4. 동파법': 'DSS', '5. 떨사오팔': '0458', '6. 종사종팔3': 'Jong Jong'}

# ==============================================================================
# [1] Streamlit UI 구성 및 입력값 처리
# ==============================================================================
# --- 🌟 설정 대시보드 (1/3) ---
st.header("🎛️ 설정 대시보드")
st.markdown("---")

col_strat, col_ticker, col_method = st.columns(3)
with col_strat:
    s_name = st.selectbox("📌 매매전략", list(STRATEGY_DB.keys()), index=0)
with col_ticker:
    ticker = st.text_input("📈 종목코드", value="SOXL").strip().upper()
with col_method:
    method = st.selectbox("⚖️ 매수방식", ['정액매수 (분모=종가)', '정수매수 (분모=목표가)'], index=1)

config = STRATEGY_DB[s_name]

col_seed, col_split, col_cycle = st.columns(3)
with col_seed:
    seed = st.number_input("💰 초기자본($)", value=40000, step=1000)
with col_split:
    split = st.number_input("🔢 분할수", value=config['split'], min_value=1, step=1)
with col_cycle:
    cycle = st.number_input("🔄 갱신주기(일)", value=config['cycle'], min_value=1, step=1)

# --- 🌟 설정 대시보드 (2/3) ---
col_profit, col_loss, col_start, col_end = st.columns(4)
with col_profit:
    p_rate = st.number_input("🔺 이익복리(%)", value=config['profit'], step=0.1)
with col_loss:
    l_rate = st.number_input("🔻 손실복리(%)", value=config['loss'], step=0.1)
with col_start:
    start_d = st.date_input("📅 시작일", value=date(2025, 1, 1))
with col_end:
    end_d = st.date_input("🏁 종료일", value=datetime.now().date())

# --- 🌟 전략 파라미터 튜닝 (3/3) ---
st.subheader("⚙️ 전략 파라미터 튜닝")

modes = ['Turbo', 'Sports', 'Comfort', 'Eco']
params_labels = ['매수율(%)', '익절율(%)', 'SL(상단)', 'SL(중단)', 'SL(하단)']
param_keys = ['Buy', 'Sell', 'SL_H', 'SL_M', 'SL_L']

# Streamlit expander를 사용하여 깔끔하게 정리
with st.expander("세부 매매 파라미터 및 비중 설정 (클릭)", expanded=False):
    # 파라미터 매트릭스 입력
    st.markdown("##### 모드별 매수/익절율 및 손절일 설정 (단위: % / 일)")
    cols_params = st.columns(len(modes) + 1)
    cols_params[0].markdown('**모드**')
    for idx, mode in enumerate(modes):
        cols_params[idx + 1].markdown(f'**{mode}**')

    custom_rules = {}
    custom_sl_matrix = {}

    for r_idx, label in enumerate(params_labels):
        p_key = param_keys[r_idx]
        cols = st.columns(len(modes) + 1)
        cols[0].markdown(f"**{label}**")
        
        current_vals = {m: (config['rules'][m].get(p_key.split('_')[0], 0.0) if p_key in ['Buy', 'Sell'] else config['sl_matrix'][m][r_idx - 2])
                        for m in modes if m in config['rules']}

        for c_idx, mode in enumerate(modes):
            default_val = current_vals.get(mode, 0.0)
            key_id = f"{s_name}_{mode}_{p_key}"
            
            if 'SL' in p_key:
                value = cols[c_idx + 1].number_input('', value=int(default_val), key=key_id, min_value=0, step=1, label_visibility="collapsed")
            else:
                value = cols[c_idx + 1].number_input('', value=float(default_val), key=key_id, step=0.1, label_visibility="collapsed")
            
            # 결과 저장
            if p_key in ['Buy', 'Sell']:
                custom_rules.setdefault(mode, {})[p_key] = value * 0.01  # %를 소수로 변환
            else:
                custom_sl_matrix.setdefault(mode, [0, 0, 0])[r_idx - 2] = int(value)

    # 분할별 비중 설정
    st.markdown("##### 분할별 비중 (1회 투입금 배수)")
    custom_weights = {}
    cols_weights = st.columns(5)
    for i in range(1, 11):
        default_weight = config['weights'].get(i, 0.0)
        
        if i <= split:
            w = cols_weights[(i - 1) % 5].number_input(f"{i}차 비중", value=default_weight, key=f"weight_{i}", step=0.1)
            custom_weights[i] = w
        else:
            custom_weights[i] = 0.0


# --- Run Button ---
if st.button("✨ 시뮬레이션 시작 (RUN)", type="primary", use_container_width=True):
    # 실행
    # [Error 방지] Streamlit은 모든 코드를 실행하므로, 버튼 클릭 시에만 로직이 실행되도록 함수로 감쌈.
    st.session_state['run_sim'] = True
else:
    if 'run_sim' not in st.session_state:
        st.session_state['run_sim'] = False

# ==============================================================================
# [2] 시뮬레이션 엔진 (핵심 로직)
# ==============================================================================
# 캐싱 적용: 데이터 로드 시간을 단축하여 Streamlit의 효율성을 높임.
@st.cache_data
def get_data(ticker_input, start_date, end_date):
    buffer_date = start_date - timedelta(weeks=60)
    qqq = yf.download("QQQ", start=buffer_date, end=end_date + timedelta(days=1), auto_adjust=False, progress=False)
    target = yf.download(ticker_input, start=buffer_date, end=end_date + timedelta(days=1), auto_adjust=False, progress=False)
    if isinstance(qqq.columns, pd.MultiIndex): qqq.columns = qqq.columns.get_level_values(0)
    if isinstance(target.columns, pd.MultiIndex): target.columns = target.columns.get_level_values(0)
    return qqq, target

def run_simulation_logic():
    # ------------------------------------------------
    # 🚨🚨🚨 이 부분의 로직은 코랩 버전과 100% 동일합니다. 🚨🚨🚨
    # ------------------------------------------------
    st_name = s_name
    st_name_en = STRATEGY_EN_MAP.get(st_name, st_name)
    ticker_input = ticker
    seed_input = float(seed)
    split_input = int(split)
    start_date = start_d
    end_date = end_d
    method_input = method
    profit_rate = float(p_rate) * 0.01
    loss_rate = float(l_rate) * 0.01
    update_cycle = int(cycle)

    try:
        qqq, target = get_data(ticker_input, start_date, end_date)
        if qqq.empty or target.empty:
            st.error("데이터 로드 실패 또는 종목 코드가 잘못되었습니다.")
            return

        # --- RSI 및 모드 계산 (로직 동일) ---
        q_weekly = qqq['Close'].resample('W-FRI').last().to_frame()
        delta = q_weekly['Close'].diff()
        up = delta.clip(lower=0).rolling(14).mean(); down = (-1 * delta.clip(upper=0)).rolling(14).mean()
        rs = up / down.replace(0, np.nan); q_weekly['wRSI'] = 100 - (100 / (1 + rs))
        q_weekly['RSI_1'] = q_weekly['wRSI'].shift(1); q_weekly['RSI_2'] = q_weekly['wRSI'].shift(2)

        modes_std, modes_dp = [], []; p_std, p_dp = "Comfort", "Eco"
        # [모드 판별 로직]
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

        d_delta = target['Close'].diff()
        up2 = d_delta.clip(lower=0).rolling(14).mean(); down2 = (-1 * d_delta.clip(upper=0)).abs().rolling(14).mean()
        rs2 = up2 / down2.replace(0, np.nan); target['dRSI'] = 100 - (100 / (1 + rs2))
        target['Change'] = target['Close'].pct_change() * 100
        target['wRSI'] = q_weekly['wRSI'].reindex(target.index, method='bfill')
        target['Mode_Std'] = q_weekly['Mode_Std'].reindex(target.index, method='bfill').fillna("Comfort")
        target['Mode_Dongpa'] = q_weekly['Mode_Dongpa'].reindex(target.index, method='bfill').fillna("Eco")
        target['Mode'] = target['Mode_Dongpa'] if config['mode_logic'] == 'Dongpa' else target['Mode_Std']
        
        # --- 파라미터 적용 (커스텀 룰 반영) ---
        def get_params(row):
            m = row['Mode']; dr = row['dRSI']
            if not config['use_mode']: m = "Comfort"
            # UI에서 설정된 값 사용 (global scope에 있는 custom_rules, custom_sl_matrix 사용)
            rs_local = custom_rules.get(m, {'Buy': 0.0, 'Sell': 0.0}) 
            sl_list = custom_sl_matrix.get(m, [15, 17, 20])

            sl = sl_list[1]
            if pd.notnull(dr):
                if dr >= 58: sl = sl_list[0]
                elif dr <= 40: sl = sl_list[2]
            return pd.Series([rs_local.get("Buy", 0.0), rs_local.get("Sell", 0.0), sl])

        target[['Buy_Rate', 'Sell_Rate', 'SL_Days']] = target.apply(get_params, axis=1)
        target['Prev_Close'] = target['Close'].shift(1)
        target['Target_Price'] = target['Prev_Close'] * (1 + target['Buy_Rate'])

        df = target.loc[start_date:end_date].copy()
        if df.empty:
            st.error("해당 기간의 데이터가 없습니다.")
            return

        # --- 시뮬레이션 초기화 및 루프 (로직 동일) ---
        df['Split_Count'] = split_input; df['Real_Split'] = 0; df['Split_Weight'] = 0.0
        df['1_Time_Input'] = 0.0; df['Input_Asset'] = float(seed_input); df['Update_Amt'] = 0.0
        df['Is_Buy'] = False; df['Actual_Buy_Price'] = 0.0; df['Buy_Vol'] = 0
        df['Sell_Target_Price'] = np.nan; df['TP_Price'] = np.nan; df['TP_Date'] = None
        df['SL_Price'] = np.nan; df['SL_Date'] = None; df['Status'] = ""; df['Daily_PnL'] = 0.0
        df['Daily_Sell_Amt'] = 0.0; df['Total_Buy_Amt'] = 0.0; df['Total_Eval_Amt'] = 0.0
        df['Total_Deposit'] = 0.0; df['Total_Asset'] = 0.0

        current_real_cash = float(seed_input); current_input_asset = float(seed_input)
        period_net_accum = 0.0; days_counter = 0; portfolio = []; current_split = 0
        WEIGHTS = custom_weights # UI에서 받은 비중 사용
        trade_win_cnt = 0; trade_loss_cnt = 0; gross_profit = 0.0; gross_loss = 0.0

        def format_short_date(dt): return dt.strftime("%y/%m/%d").replace("/0", "/")

        for i in range(len(df)):
            days_counter += 1; update_amount = 0.0
            if days_counter > update_cycle:
                update_amount = period_net_accum * profit_rate if period_net_accum > 0 else period_net_accum * loss_rate
                current_input_asset += update_amount; days_counter = 1; period_net_accum = 0.0
            df.iloc[i, df.columns.get_loc('Input_Asset')] = current_input_asset
            df.iloc[i, df.columns.get_loc('Update_Amt')] = update_amount
            curr_date = df.index[i].date(); curr_close = float(df['Close'].iloc[i])

            target_split_level = current_split + 1
            weight = WEIGHTS.get(target_split_level, 0.0)
            if target_split_level > split_input: weight = 0.0
            df.iloc[i, df.columns.get_loc('Split_Weight')] = weight
            
            one_time_input = (current_input_asset / split_input) * weight
            if current_real_cash < 0: one_time_input = 0.0
            else: one_time_input = min(one_time_input, current_real_cash)
            df.iloc[i, df.columns.get_loc('1_Time_Input')] = one_time_input

            daily_status = []; new_portfolio = []; daily_pnl_accum = 0.0; daily_sell_accum = 0.0; sell_occurred_today = 0
            for item in portfolio:
                if item['type'] == 'HOLD': new_portfolio.append(item); continue
                if item['sell_date'] == curr_date:
                    trade_profit = (item['sell_price'] - item['price']) * item['qty']
                    sell_amount = item['sell_price'] * item['qty']
                    current_real_cash += sell_amount
                    daily_pnl_accum += trade_profit; daily_sell_accum += sell_amount; period_net_accum += trade_profit
                    if trade_profit > 0: trade_win_cnt += 1; gross_profit += trade_profit
                    elif trade_profit < 0: trade_loss_cnt += 1; gross_loss += abs(trade_profit)
                    label = "익절" if item['type'] == 'TP' else "손절"
                    if label not in daily_status: daily_status.append(label)
                    sell_occurred_today += 1
                else: new_portfolio.append(item)
            portfolio = new_portfolio; current_split -= sell_occurred_today
            if current_split < 0: current_split = 0

            prev_close = df['Prev_Close'].iloc[i]
            if pd.notnull(prev_close):
                target_price = float(df['Target_Price'].iloc[i])
                if curr_close <= target_price:
                    df.iloc[i, df.columns.get_loc('Is_Buy')] = True; df.iloc[i, df.columns.get_loc('Actual_Buy_Price')] = curr_close
                    buy_vol = 0
                    if one_time_input > 0:
                        denominator = curr_close if '정액매수' in method_input else target_price
                        buy_vol = math.floor(one_time_input / denominator)
                        max_buyable = math.floor(current_real_cash / curr_close)
                        buy_vol = min(buy_vol, max_buyable)
                        current_real_cash -= (buy_vol * curr_close)
                    if target_split_level <= split_input:
                        df.iloc[i, df.columns.get_loc('Buy_Vol')] = buy_vol
                        if "매수" not in daily_status: daily_status.append("매수")
                        current_split += 1
                        sell_rate = float(df['Sell_Rate'].iloc[i]); sl_days = int(df['SL_Days'].iloc[i])
                        sell_target = curr_close * (1 + sell_rate)
                        df.iloc[i, df.columns.get_loc('Sell_Target_Price')] = sell_target
                        start_idx = i + 1; target_sl_idx = i + sl_days; end_idx = min(target_sl_idx + 1, len(df))
                        sell_date = None; sell_price_res = 0.0; sell_type_res = 'HOLD'
                        if start_idx < len(df):
                            future_window = df.iloc[start_idx:end_idx]; hit_mask = future_window['Close'] >= sell_target
                            if hit_mask.any():
                                sell_idx = hit_mask.idxmax(); hit_row = df.loc[sell_idx]
                                df.iloc[i, df.columns.get_loc('TP_Price')] = float(hit_row['Close'])
                                df.iloc[i, df.columns.get_loc('TP_Date')] = format_short_date(sell_idx)
                                sell_date = sell_idx.date(); sell_price_res = float(hit_row['Close']); sell_type_res = 'TP'
                            elif sl_days > 0 and target_sl_idx < len(df):
                                sell_idx = df.index[target_sl_idx]; last_row = df.loc[sell_idx]
                                df.iloc[i, df.columns.get_loc('SL_Price')] = float(last_row['Close'])
                                df.iloc[i, df.columns.get_loc('SL_Date')] = format_short_date(sell_idx)
                                sell_date = sell_idx.date(); sell_price_res = float(last_row['Close']); sell_type_res = 'SL'
                            portfolio.append({'qty': int(buy_vol), 'price': float(curr_close), 'sell_date': sell_date, 'sell_price': float(sell_price_res), 'type': sell_type_res})

            total_buy_amt = sum([item['qty'] * item['price'] for item in portfolio])
            total_eval_amt = sum([item['qty'] * curr_close for item in portfolio])
            total_asset = current_real_cash + total_eval_amt

            df.iloc[i, df.columns.get_loc('Status')] = ",".join(daily_status)
            df.iloc[i, df.columns.get_loc('Daily_Sell_Amt')] = daily_sell_accum
            df.iloc[i, df.columns.get_loc('Daily_PnL')] = daily_pnl_accum
            df.iloc[i, df.columns.get_loc('Total_Buy_Amt')] = total_buy_amt
            df.iloc[i, df.columns.get_loc('Total_Eval_Amt')] = total_eval_amt
            df.iloc[i, df.columns.get_loc('Total_Deposit')] = current_real_cash
            df.iloc[i, df.columns.get_loc('Total_Asset')] = total_asset
            df.iloc[i, df.columns.get_loc('Real_Split')] = current_split

        # --- Metrics Calculation ---
        df['Accum_Return'] = (df['Total_Asset'] - float(seed_input)) / float(seed_input) * 100
        df['Peak_Asset'] = df['Total_Asset'].cummax()
        df['DD'] = (df['Total_Asset'] - df['Peak_Asset']) / df['Peak_Asset'] * 100

        final_asset = float(df['Total_Asset'].iloc[-1])
        total_return = (final_asset - seed_input) / seed_input * 100
        mdd = float(df['DD'].min())
        total_days = (df.index[-1] - df.index[0]).days
        years = total_days / 365.25
        cagr = ((final_asset / seed_input) ** (1 / years) - 1) * 100 if (years > 0 and final_asset > 0) else 0.0

        total_trades = trade_win_cnt + trade_loss_cnt
        win_rate = (total_trades > 0 and trade_win_cnt / total_trades * 100) or 0.0
        gross_profit = gross_profit if trade_win_cnt > 0 else 0.0
        gross_loss = gross_loss if trade_loss_cnt > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.99 if gross_profit > 0 else 0.0)
        avg_win = (gross_profit / trade_win_cnt) if trade_win_cnt > 0 else 0.0
        avg_loss = (gross_loss / trade_loss_cnt) if trade_loss_cnt > 0 else 0.0

        # --- 📊 Streamlit Dashboard (Metric Cards) ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Return", f"{total_return:+.2f}%", f"CAGR {cagr:.1f}%")
        k2.metric("Final Asset", f"${final_asset:,.0f}", f"Seed: ${seed_input:,.0f}")
        k3.metric("Max Drawdown", f"{mdd:.2f}%", "Risk Tolerance")
        k4.metric("Win Rate", f"{win_rate:.1f}%", f"W:{trade_win_cnt} | L:{trade_loss_cnt}")
        k5.metric("Profit Factor", f"{profit_factor:.2f}", f"Avg W ${avg_win:,.0f}")

        # --- 🖼️ Matplotlib Chart ---
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3) 

        ax1 = plt.subplot(gs[0])
        line1 = ax1.plot(df.index, df['Total_Asset'], label='Total Asset', color='#e74c3c', linewidth=2)
        ax1.fill_between(df.index, df['Total_Asset'], df['Total_Asset'].min(), color='#e74c3c', alpha=0.05)
        
        ax1_twin = ax1.twinx()
        line2 = ax1_twin.plot(df.index, df['Close'], label='Price', color='#95a5a6', alpha=0.6, linewidth=1, linestyle='--')
        
        tp_df = df[df['Status'].str.contains('익절', na=False)]; sl_df = df[df['Status'].str.contains('손절', na=False)]
        ax1.scatter(tp_df.index, tp_df['Total_Asset'], marker='^', color='#e74c3c', s=60, zorder=5)
        ax1.scatter(sl_df.index, sl_df['Total_Asset'], marker='v', color='#2980b9', s=60, zorder=5)
        
        ax1.set_ylabel('Asset ($)', fontsize=11, fontweight='bold', color='#e74c3c')
        ax1_twin.set_ylabel('Stock Price ($)', fontsize=11, color='#95a5a6')
        ax1.set_title(f"🚀 Asset Growth & Price Action ({ticker_input}) - {st_name_en}", fontsize=14, fontweight='bold', pad=10)
        
        lines = line1 + line2; labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.9, shadow=True)
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(df.index, df['DD'], color='#2980b9', linewidth=1)
        ax2.fill_between(df.index, df['DD'], 0, color='#2980b9', alpha=0.2)
        ax2.set_title("Drawdown", fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.5)

        ax3 = plt.subplot(gs[2], sharex=ax1)
        colors = ['#2980b9' if v < 0 else '#e74c3c' for v in df['Daily_PnL']]
        ax3.bar(df.index, df['Daily_PnL'], color=colors, alpha=0.8)
        ax3.set_title("Daily PnL", fontsize=11, fontweight='bold')
        ax3.grid(True, linestyle=':', alpha=0.5)
        
        st.pyplot(fig)

        # --- 📄 상세 테이블 ---
        st.subheader("📋 일별 상세 거래 내역")
        cols = ['Close', 'Change', 'wRSI', 'dRSI', 'Mode', 'Buy_Rate', 'Sell_Rate', 'SL_Days',
                'Real_Split', 'Input_Asset', 'Split_Count', 'Split_Weight', '1_Time_Input', 'Update_Amt', 
                'Target_Price', 'Actual_Buy_Price', 'Buy_Vol', 'Sell_Target_Price', 'TP_Price', 'TP_Date', 'SL_Price', 'SL_Date', 
                'Status', 'Daily_Sell_Amt', 'Daily_PnL', 'Total_Buy_Amt', 'Total_Eval_Amt', 'Total_Deposit', 'Total_Asset', 'Accum_Return', 'DD']
        df_disp = df[cols].copy()
        col_map = {
            'Close': '종가', 'Change': '등락(%)', 'Mode': '모드', 'Buy_Rate': '매수율', 'Sell_Rate': '익절율', 'SL_Days': '손절(일)',
            'Real_Split': '분할', 'Input_Asset': '투입자산', 'Split_Count': '설정분할', 'Split_Weight': '비중', '1_Time_Input': '1회투입',
            'Update_Amt': '갱신금', 'Target_Price': '매수목표', 'Actual_Buy_Price': '실매수', 'Buy_Vol': '매수량',
            'Sell_Target_Price': '매도목표', 'TP_Price': '익절가', 'TP_Date': '익절일', 'SL_Price': '손절가', 'SL_Date': '손절일',
            'Status': '상태', 'Daily_Sell_Amt': '매도액', 'Daily_PnL': '손익', 'Total_Buy_Amt': '매수총액',
            'Total_Eval_Amt': '평가총액', 'Total_Deposit': '예수금', 'Total_Asset': '자산', 'Accum_Return': '수익률', 'DD': 'DD'
        }
        df_disp.rename(columns=col_map, inplace=True)
        df_disp.index = df_disp.index.strftime('%Y-%m-%d')
        st.dataframe(df_disp, use_container_width=True)


    except Exception as e:
        st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")

# --- Streamlit 실행 (버튼 클릭 여부에 따라 로직 실행) ---
if st.session_state.get('run_sim'):
    run_simulation_logic()

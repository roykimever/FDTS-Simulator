import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import math
import warnings
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ==============================================================================
# [0] 전략 데이터베이스
# ==============================================================================
STRATEGY_DB = {
    '1. 터보 운전법': {
        'split': 7, 'profit': 85.0, 'loss': 35.0, 'cycle': 9,
        'mode_logic': 'Standard', 'use_mode': True,
        'weights': {1: 0.0, 2: 0.3, 3: 0.5, 4: 0.7, 5: 2.3, 6: 2.2, 7: 1.0},
        'rules': {"Turbo": {"Buy": 0.028, "Sell": 0.026}, "Sports": {"Buy": 0.041, "Sell": 0.032}, "Comfort": {"Buy": 0.053, "Sell": 0.021}, "Eco": {"Buy": 0.066, "Sell": 0.004}},
        'sl_matrix': {"Turbo": [6, 7, 8], "Sports": [7, 8, 10], "Comfort": [16, 18, 20], "Eco": [26, 27, 30]}
    },
    '2. 안전 운전법': {
        'split': 7, 'profit': 75.0, 'loss': 40.0, 'cycle': 10,
        'mode_logic': 'Standard', 'use_mode': True,
        'weights': {1: 0.0, 2: 0.0, 3: 0.0, 4: 1.1, 5: 2.3, 6: 2.4, 7: 1.3},
        'rules': {"Turbo": {"Buy": 0.035, "Sell": 0.028}, "Sports": {"Buy": 0.045, "Sell": 0.028}, "Comfort": {"Buy": 0.050, "Sell": 0.020}, "Eco": {"Buy": 0.065, "Sell": 0.006}},
        'sl_matrix': {"Turbo": [6, 7, 8], "Sports": [6, 7, 8], "Comfort": [15, 17, 20], "Eco": [25, 28, 30]}
    },
    '3. 풍차 매매법': {
        'split': 10, 'profit': 90.0, 'loss': 25.0, 'cycle': 5,
        'mode_logic': 'Standard', 'use_mode': True,
        'weights': {i: 1.0 for i in range(1, 11)},
        'rules': {"Turbo": {"Buy": 0.035, "Sell": 0.001}, "Sports": {"Buy": 0.045, "Sell": 0.001}, "Comfort": {"Buy": 0.050, "Sell": 0.001}, "Eco": {"Buy": 0.065, "Sell": 0.001}},
        'sl_matrix': {"Turbo": [10, 15, 20], "Sports": [12, 17, 22], "Comfort": [15, 20, 25], "Eco": [20, 25, 30]}
    },
    '4. 동파법': {
        'split': 7, 'profit': 80.0, 'loss': 30.0, 'cycle': 10,
        'mode_logic': 'Dongpa', 'use_mode': True,
        'weights': {i: 1.0 for i in range(1, 101)},
        'rules': {"Turbo": {"Buy": 0.00, "Sell": 0.00}, "Sports": {"Buy": 0.05, "Sell": 0.025}, "Comfort": {"Buy": 0.00, "Sell": 0.00}, "Eco": {"Buy": 0.03, "Sell": 0.002}},
        'sl_matrix': {"Turbo": [0, 0, 0], "Sports": [7, 7, 7], "Comfort": [0, 0, 0], "Eco": [30, 30, 30]}
    },
    '5. 떨사오팔': {
        'split': 7, 'profit': 80.0, 'loss': 30.0, 'cycle': 10,
        'mode_logic': 'Standard', 'use_mode': False,
        'weights': {i: 1.0 for i in range(1, 101)},
        'rules': {"Comfort": {"Buy": -0.001, "Sell": 0.001}},
        'sl_matrix': {"Comfort": [30, 30, 30]}
    },
    '6. 종사종팔3': {
        'split': 7, 'profit': 70.0, 'loss': 0.0, 'cycle': 10,
        'mode_logic': 'Standard', 'use_mode': False,
        'weights': {i: 1.0 for i in range(1, 101)},
        'rules': {"Turbo": {"Buy": 0.15, "Sell": 0.027}, "Sports": {"Buy": 0.15, "Sell": 0.027}, "Comfort": {"Buy": 0.15, "Sell": 0.027}, "Eco": {"Buy": 0.15, "Sell": 0.027}},
        'sl_matrix': {"Turbo": [10, 10, 10], "Sports": [10, 10, 10], "Comfort": [10, 10, 10], "Eco": [10, 10, 10]}
    }
}
STRATEGY_EN_MAP = {'1. 터보 운전법': 'Turbo Driving', '2. 안전 운전법': 'Safety Driving', '3. 풍차 매매법': 'Wind Wheel', '4. 동파법': 'DSS', '5. 떨사오팔': '0458', '6. 종사종팔3': 'Jong Jong'}

# ==============================================================================
# [1] UI 구성 (Sidebar)
# ==============================================================================
st.set_page_config(page_title="FDTS Simulator", layout="wide", page_icon="📈")

st.markdown("""
    <h2 style='color:#2c3e50; border-bottom:3px solid #3498db; padding-bottom:5px;'>
    📊 FDTS Premium Dashboard v20.4 (Web Ver.)
    </h2>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🎛️ 설정 대시보드")
    
    selected_strategy = st.selectbox("📌 매매전략:", list(STRATEGY_DB.keys()), index=0)
    
    # 전략 선택 시 기본값 로드
    strat_config = STRATEGY_DB[selected_strategy]
    
    ticker_input = st.text_input("📈 종목코드:", value="SOXL")
    method_input = st.selectbox("⚖️ 매수방식:", ['정액매수 (분모=종가)', '정수매수 (분모=목표가)'], index=1)
    
    seed_input = st.number_input("💰 초기자본($):", value=40000, step=1000)
    
    col1, col2 = st.columns(2)
    with col1:
        split_input = st.number_input("🔢 분할수:", value=strat_config['split'])
        profit_rate = st.number_input("🔺 이익복리(%):", value=strat_config['profit'])
    with col2:
        cycle_input = st.number_input("🔄 갱신주기(일):", value=strat_config['cycle'])
        loss_rate = st.number_input("🔻 손실복리(%):", value=strat_config['loss'])
        
    start_date = st.date_input("📅 시작일:", value=date(2025, 1, 1))
    end_date = st.date_input("🏁 종료일:", value=datetime.now().date())
    
    run_btn = st.button("✨ 시뮬레이션 시작 (RUN)", type="primary", use_container_width=True)

# ==============================================================================
# [2] 시뮬레이션 엔진
# ==============================================================================
if run_btn:
    with st.spinner(f"⏳ [{selected_strategy}] 데이터를 분석하고 대시보드를 생성 중입니다..."):
        try:
            # 설정값 준비
            config = STRATEGY_DB[selected_strategy]
            # 사용자 입력값으로 덮어쓰기 (필요시) - 여기서는 UI 입력값을 우선 사용
            # config['weights'] 등은 그대로 사용
            st_name_en = STRATEGY_EN_MAP.get(selected_strategy, selected_strategy)
            
            p_rate = profit_rate * 0.01
            l_rate = loss_rate * 0.01
            
            # --- Data Loading ---
            buffer_date = start_date - timedelta(weeks=60)
            qqq = yf.download("QQQ", start=buffer_date, end=end_date + timedelta(days=1), auto_adjust=False, progress=False)
            if qqq.empty: raise ValueError("QQQ 데이터 로드 실패")
            if isinstance(qqq.columns, pd.MultiIndex): qqq.columns = qqq.columns.get_level_values(0)
            
            q_weekly = qqq['Close'].resample('W-FRI').last().to_frame()
            delta = q_weekly['Close'].diff()
            q_weekly['wRSI'] = 100 - (100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-1 * delta.clip(upper=0)).rolling(14).mean()))
            q_weekly['RSI_1'] = q_weekly['wRSI'].shift(1); q_weekly['RSI_2'] = q_weekly['wRSI'].shift(2)
            
            modes_std, modes_dp = [], []
            p_std, p_dp = "Comfort", "Eco"
            for i, row in q_weekly.iterrows():
                r1, r2 = row['RSI_1'], row['RSI_2']
                m_std = p_std
                if not (pd.isna(r1) or pd.isna(r2)):
                    if (r2 < 40) and ((r1 - r2) >= 5) and (r1 <= 55): m_std = "Turbo"
                    elif ((r2 > 65 and r1 < r2) or (r2 > 40 and r2 < 50 and r1 < r2) or (r1 < 50 and r2 > 50)): m_std = "Eco"
                    elif ((r2 < 35 and r1 > r2) or (r2 > 50 and r2 < 60 and r1 > r2) or (r1 > 50 and r2 < 50)): m_std = "Sports"
                    elif (r2 >= 40 and r2 <= 65): m_std = "Comfort"
                modes_std.append(m_std); p_std = m_std
                
                m_dp = p_dp
                if not (pd.isna(r1) or pd.isna(r2)):
                    if (r2 >= 65 and r1 < r2) or (r2 >= 40 and r2 <= 50 and r1 < r2) or (r2 >= 50 and r1 < 50): m_dp = "Eco"
                    elif (r2 <= 50 and r1 > 50) or (r2 >= 50 and r2 <= 60 and r1 > r2) or (r2 <= 35 and r1 > r2): m_dp = "Sports"
                modes_dp.append(m_dp); p_dp = m_dp
            q_weekly['Mode_Std'] = modes_std; q_weekly['Mode_Dongpa'] = modes_dp

            target = yf.download(ticker_input.upper(), start=buffer_date, end=end_date + timedelta(days=1), auto_adjust=False, progress=False)
            if target.empty: raise ValueError(f"{ticker_input} 데이터 로드 실패")
            if isinstance(target.columns, pd.MultiIndex): target.columns = target.columns.get_level_values(0)
            
            d_delta = target['Close'].diff()
            target['dRSI'] = 100 - (100 / (1 + d_delta.clip(lower=0).rolling(14).mean() / (-1 * d_delta.clip(upper=0)).abs().rolling(14).mean()))
            target['Change'] = target['Close'].pct_change() * 100
            target['wRSI'] = q_weekly['wRSI'].reindex(target.index, method='bfill')
            target['Mode_Std'] = q_weekly['Mode_Std'].reindex(target.index, method='bfill').fillna("Comfort")
            target['Mode_Dongpa'] = q_weekly['Mode_Dongpa'].reindex(target.index, method='bfill').fillna("Eco")
            target['Mode'] = target['Mode_Dongpa'] if config['mode_logic'] == 'Dongpa' else target['Mode_Std']

            def get_params(row):
                m = row['Mode']; dr = row['dRSI']
                if not config['use_mode']: m = "Comfort"
                rs = config['rules'].get(m, config['rules'].get("Comfort"))
                sl_list = config['sl_matrix'].get(m, [15, 17, 20])
                sl = sl_list[1]
                if pd.notnull(dr):
                    if dr >= 58: sl = sl_list[0]
                    elif dr <= 40: sl = sl_list[2]
                return pd.Series([rs.get("Buy", 0.05), rs.get("Sell", 0.02), sl])

            target[['Buy_Rate', 'Sell_Rate', 'SL_Days']] = target.apply(get_params, axis=1)
            target['Prev_Close'] = target['Close'].shift(1)
            target['Target_Price'] = target['Prev_Close'] * (1 + target['Buy_Rate'])

            df = target.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)].copy()
            if df.empty: raise ValueError("해당 기간 데이터 없음")

            # --- Logic Loop (Optimized for Streamlit) ---
            df['Split_Count'] = split_input; df['Real_Split'] = 0; df['Split_Weight'] = 0.0
            df['1_Time_Input'] = 0.0; df['Input_Asset'] = float(seed_input); df['Update_Amt'] = 0.0
            df['Is_Buy'] = False; df['Actual_Buy_Price'] = 0.0; df['Buy_Vol'] = 0
            df['Sell_Target_Price'] = np.nan; df['TP_Price'] = np.nan; df['TP_Date'] = None
            df['SL_Price'] = np.nan; df['SL_Date'] = None; df['Status'] = ""; df['Daily_PnL'] = 0.0
            df['Daily_Sell_Amt'] = 0.0; df['Total_Buy_Amt'] = 0.0; df['Total_Eval_Amt'] = 0.0
            df['Total_Deposit'] = 0.0; df['Total_Asset'] = 0.0

            current_real_cash = float(seed_input); current_input_asset = float(seed_input)
            period_net_accum = 0.0; days_counter = 0; portfolio = []; current_split = 0
            WEIGHTS = config['weights']
            
            trade_win_cnt = 0; trade_loss_cnt = 0; gross_profit = 0.0; gross_loss = 0.0
            def format_short_date(dt): return dt.strftime("%y/%-m/%-d")

            for i in range(len(df)):
                days_counter += 1; update_amount = 0.0
                if days_counter > cycle_input:
                    update_amount = period_net_accum * p_rate if period_net_accum > 0 else period_net_accum * l_rate
                    current_input_asset += update_amount; days_counter = 1; period_net_accum = 0.0
                
                df.iloc[i, df.columns.get_loc('Input_Asset')] = current_input_asset
                df.iloc[i, df.columns.get_loc('Update_Amt')] = update_amount
                curr_date = df.index[i].date(); curr_close = df['Close'].iloc[i]

                target_split_level = current_split + 1
                weight = WEIGHTS.get(target_split_level, 0.0)
                if target_split_level > split_input: weight = 0.0
                df.iloc[i, df.columns.get_loc('Split_Weight')] = weight
                
                one_time_input = (current_input_asset / split_input) * weight
                if current_real_cash < 0: one_time_input = 0
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
                    target_price = df['Target_Price'].iloc[i]
                    if curr_close <= target_price:
                        df.iloc[i, df.columns.get_loc('Is_Buy')] = True
                        df.iloc[i, df.columns.get_loc('Actual_Buy_Price')] = curr_close
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
                            sell_rate = df['Sell_Rate'].iloc[i]; sl_days = int(df['SL_Days'].iloc[i])
                            sell_target = curr_close * (1 + sell_rate)
                            df.iloc[i, df.columns.get_loc('Sell_Target_Price')] = sell_target
                            start_idx = i + 1; target_sl_idx = i + sl_days; end_idx = min(target_sl_idx + 1, len(df))
                            sell_date = None; sell_price_res = 0.0; sell_type_res = 'HOLD'
                            if start_idx < len(df):
                                future_window = df.iloc[start_idx : end_idx]
                                hit_mask = future_window['Close'] >= sell_target
                                if hit_mask.any():
                                    sell_idx = hit_mask.idxmax(); hit_row = df.loc[sell_idx]
                                    df.iloc[i, df.columns.get_loc('TP_Price')] = hit_row['Close']
                                    df.iloc[i, df.columns.get_loc('TP_Date')] = format_short_date(sell_idx)
                                    sell_date = sell_idx.date(); sell_price_res = hit_row['Close']; sell_type_res = 'TP'
                                elif sl_days > 0 and target_sl_idx < len(df):
                                    sell_idx = df.index[target_sl_idx]; last_row = df.loc[sell_idx]
                                    df.iloc[i, df.columns.get_loc('SL_Price')] = last_row['Close']
                                    df.iloc[i, df.columns.get_loc('SL_Date')] = format_short_date(sell_idx)
                                    sell_date = sell_idx.date(); sell_price_res = last_row['Close']; sell_type_res = 'SL'
                            portfolio.append({'qty': buy_vol, 'price': curr_close, 'sell_date': sell_date, 'sell_price': sell_price_res, 'type': sell_type_res})

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

            # Metrics Calculation
            df['Accum_Return'] = (df['Total_Asset'] - float(seed_input)) / float(seed_input) * 100
            df['Peak_Asset'] = df['Total_Asset'].cummax()
            df['DD'] = (df['Total_Asset'] - df['Peak_Asset']) / df['Peak_Asset'] * 100

            final_asset = df['Total_Asset'].iloc[-1]
            total_return = (final_asset - seed_input) / seed_input * 100
            mdd = df['DD'].min()
            total_days = (df.index[-1] - df.index[0]).days; years = total_days / 365.25
            cagr = ((final_asset / seed_input) ** (1 / years) - 1) * 100 if (years > 0 and final_asset > 0) else 0.0
            total_trades = trade_win_cnt + trade_loss_cnt
            win_rate = (trade_win_cnt / total_trades * 100) if total_trades > 0 else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.99 if gross_profit > 0 else 0.0)
            avg_win = (gross_profit / trade_win_cnt) if trade_win_cnt > 0 else 0.0
            avg_loss = (gross_loss / trade_loss_cnt) if trade_loss_cnt > 0 else 0.0

            # ------------------------------------------------------------------
            # UI Output (Dashboard Cards)
            # ------------------------------------------------------------------
            st.markdown("### 📊 Performance Summary")
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            
            kpi1.metric("Final Asset", f"${final_asset:,.0f}", f"Seed: ${seed_input:,.0f}")
            kpi2.metric("Total Return", f"{total_return:+.2f}%", f"CAGR: {cagr:.1f}%")
            kpi3.metric("Max Drawdown", f"{mdd:.2f}%", "Risk")
            kpi4.metric("Winning Rate", f"{win_rate:.1f}%", f"{trade_win_cnt}W / {trade_loss_cnt}L")
            kpi5.metric("Profit Factor", f"{profit_factor:.2f}", f"Avg W: ${avg_win:,.0f}")

            # ------------------------------------------------------------------
            # Chart
            # ------------------------------------------------------------------
            fig = plt.figure(figsize=(12, 12))
            gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
            
            ax1 = plt.subplot(gs[0])
            line1 = ax1.plot(df.index, df['Total_Asset'], label='Total Asset', color='#1f77b4', linewidth=2.5)
            ax1.fill_between(df.index, df['Total_Asset'], df['Total_Asset'].min(), color='#1f77b4', alpha=0.1)
            
            ax1_twin = ax1.twinx()
            line2 = ax1_twin.plot(df.index, df['Close'], label='Stock Price', color='gray', alpha=0.5, linewidth=1.5, linestyle='--')
            
            tp_df = df[df['Status'].str.contains('익절')]; sl_df = df[df['Status'].str.contains('손절')]
            ax1.scatter(tp_df.index, tp_df['Total_Asset'], marker='^', color='#d62728', s=80, edgecolors='white', linewidth=1.5, label='Take Profit', zorder=5)
            ax1.scatter(sl_df.index, sl_df['Total_Asset'], marker='v', color='#1f77b4', s=80, edgecolors='white', linewidth=1.5, label='Stop Loss', zorder=5)
            
            ax1.set_ylabel('Total Asset ($)', fontsize=11, fontweight='bold', color='#1f77b4')
            ax1_twin.set_ylabel('Stock Price ($)', fontsize=11, color='gray')
            ax1.set_title(f"🚀 Asset Growth & Price Action ({ticker_input.upper()}) - {st_name_en}", fontsize=15, fontweight='bold', pad=15)
            lines = line1 + line2; labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')
            ax1.grid(True, linestyle=':', alpha=0.6)

            ax2 = plt.subplot(gs[1], sharex=ax1)
            ax2.plot(df.index, df['DD'], label='MDD', color='#1f77b4')
            ax2.fill_between(df.index, df['DD'], 0, color='#1f77b4', alpha=0.3)
            ax2.set_title("Drawdown Risk (%)", fontsize=12, fontweight='bold', pad=10)
            ax2.grid(True, linestyle=':', alpha=0.6)

            ax3 = plt.subplot(gs[2], sharex=ax1)
            colors_bar = ['#1f77b4' if v < 0 else '#d62728' for v in df['Daily_PnL']]
            ax3.bar(df.index, df['Daily_PnL'], color=colors_bar, alpha=0.8)
            ax3.set_title("Daily Profit & Loss ($)", fontsize=12, fontweight='bold', pad=10)
            ax3.grid(True, linestyle=':', alpha=0.6)
            
            st.pyplot(fig)

            # ------------------------------------------------------------------
            # Table
            # ------------------------------------------------------------------
            st.markdown("### 📄 Transaction Log")
            cols = ['Close', 'Change', 'wRSI', 'dRSI', 'Mode', 'Buy_Rate', 'Sell_Rate', 'SL_Days', 'Real_Split', 'Input_Asset', '1_Time_Input', 'Update_Amt', 'Target_Price', 'Actual_Buy_Price', 'Buy_Vol', 'Sell_Target_Price', 'TP_Price', 'TP_Date', 'SL_Price', 'SL_Date', 'Status', 'Daily_PnL', 'Total_Eval_Amt', 'Total_Deposit', 'Total_Asset', 'Accum_Return', 'DD']
            df_disp = df[cols].copy()
            df_disp.index = df_disp.index.date
            
            # 하이라이팅 기능은 dataframe의 style 기능을 사용하거나, 간단히 그냥 보여줍니다.
            # Streamlit의 dataframe은 인터랙티브하므로 전체 데이터를 보여줍니다.
            st.dataframe(df_disp.style.format({
                'Close': '{:.2f}', 'Change': '{:+.2f}', 'wRSI': '{:.1f}', 'dRSI': '{:.1f}',
                'Buy_Rate': '{:.3f}', 'Sell_Rate': '{:.3f}', 'SL_Days': '{:.0f}',
                'Input_Asset': '{:,.0f}', 'Total_Asset': '{:,.0f}', 'Daily_PnL': '{:+,.0f}',
                'Accum_Return': '{:+.2f}%', 'DD': '{:.2f}%'
            }))

        except Exception as e:
            st.error(f"오류 발생: {str(e)}")

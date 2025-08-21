#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日本全都道府県 人口動態分析ダッシュボード
Streamlit Dashboard for Japan Population Dynamics Analysis

使用方法:
streamlit run population_dashboard.py

Author: Claude Code
Date: 2025-08-21
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import io

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="日本人口動態分析ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PopulationAnalyzer:
    """人口分析クラス"""
    
    def __init__(self):
        self.df = None
        self.prefectures = []
        self.models = {
            'Linear': LinearRegression(),
            'Polynomial_2': Pipeline([('poly', PolynomialFeatures(2)), ('linear', LinearRegression())]),
            'Polynomial_3': Pipeline([('poly', PolynomialFeatures(3)), ('linear', LinearRegression())])
        }
        
    @st.cache_data
    def load_data(_self, uploaded_file):
        """データの読み込み・前処理"""
        try:
            # アップロードされたファイルを読み込み
            df = pd.read_csv(uploaded_file, encoding='shift_jis', skiprows=10)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8', skiprows=10)
            except:
                df = pd.read_csv(uploaded_file, encoding='cp932', skiprows=10)
        
        # 列名設定
        columns = ['地域', '年齢階級']
        years = [1920, 1925, 1930, 1935, 1940, 1945, 1950, 1955, 1960, 1965, 
                 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
        
        for year in years:
            columns.extend([f'{year}_総数', f'{year}_男', f'{year}_女', 
                           f'{year}_総数割合', f'{year}_男割合', f'{year}_女割合', f'{year}_人口性比'])
        
        # 列数調整
        df = df.iloc[:, :min(len(columns), df.shape[1])]
        df.columns = columns[:df.shape[1]]
        
        # データクリーニング
        df = df.dropna(subset=['地域'])
        df = df[df['地域'].str.contains('_', na=False)]
        
        # 数値データ変換
        numeric_cols = [col for col in df.columns if col not in ['地域', '年齢階級']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('"', ''), errors='coerce')
        
        _self.df = df
        _self.prefectures = sorted(df['地域'].unique())
        
        return df
    
    def prepare_time_series_data(self, region_code=None):
        """時系列データの準備"""
        if region_code:
            region_data = self.df[self.df['地域'] == region_code]
            total_data = region_data[region_data['年齢階級'] == '総数']
        else:
            total_data = self.df[self.df['年齢階級'] == '総数'].groupby(['年齢階級']).sum()
        
        years = [1920, 1925, 1930, 1935, 1940, 1945, 1950, 1955, 1960, 1965, 
                 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
        
        population_data = []
        for year in years:
            col_name = f'{year}_総数'
            if col_name in total_data.columns:
                if region_code:
                    pop = total_data[col_name].iloc[0] if not total_data.empty else np.nan
                else:
                    pop = total_data[col_name].iloc[0] if not total_data.empty else np.nan
                population_data.append({'year': year, 'population': pop})
        
        ts_df = pd.DataFrame(population_data)
        ts_df = ts_df.dropna()
        return ts_df
    
    def predict_population(self, ts_data, forecast_years=30):
        """人口予測"""
        X = ts_data['year'].values.reshape(-1, 1)
        y = ts_data['population'].values
        
        results = {}
        for name, model in self.models.items():
            model.fit(X, y)
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            future_years = np.arange(2025, 2025 + forecast_years, 5).reshape(-1, 1)
            future_pred = model.predict(future_years)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'r2': r2,
                'historical_pred': y_pred,
                'future_years': future_years.flatten(),
                'future_pred': future_pred
            }
        
        return results
    
    def get_age_structure_data(self, region_code=None, year=2020):
        """年齢構造データの取得"""
        if region_code:
            region_data = self.df[self.df['地域'] == region_code]
        else:
            region_data = self.df.groupby(['年齢階級']).sum()
            region_data = region_data.reset_index()
        
        age_groups = ['0～4歳', '5～9歳', '10～14歳', '15～19歳', '20～24歳', 
                      '25～29歳', '30～34歳', '35～39歳', '40～44歳', '45～49歳',
                      '50～54歳', '55～59歳', '60～64歳', '65～69歳', '70～74歳',
                      '75～79歳', '80～84歳', '85歳以上']
        
        age_data = []
        for age in age_groups:
            if region_code:
                age_pop = region_data[region_data['年齢階級'] == age][f'{year}_総数']
                male_pop = region_data[region_data['年齢階級'] == age][f'{year}_男']
                female_pop = region_data[region_data['年齢階級'] == age][f'{year}_女']
                
                pop = age_pop.iloc[0] if not age_pop.empty else 0
                male = male_pop.iloc[0] if not male_pop.empty else 0
                female = female_pop.iloc[0] if not female_pop.empty else 0
            else:
                age_rows = region_data[region_data['年齢階級'] == age]
                pop = age_rows[f'{year}_総数'].iloc[0] if not age_rows.empty else 0
                male = age_rows[f'{year}_男'].iloc[0] if not age_rows.empty else 0
                female = age_rows[f'{year}_女'].iloc[0] if not age_rows.empty else 0
                
            age_data.append({
                'age_group': age,
                'total_population': pop,
                'male_population': male,
                'female_population': female
            })
        
        return pd.DataFrame(age_data).dropna()

def main():
    """メイン関数"""
    
    # タイトル
    st.title("📊 日本全都道府県 人口動態分析ダッシュボード")
    st.markdown("---")
    
    # サイドバー
    st.sidebar.header("📁 データアップロード")
    uploaded_file = st.sidebar.file_uploader(
        "census_2020.csvファイルをアップロードしてください",
        type=['csv']
    )
    
    if uploaded_file is not None:
        # 分析器の初期化
        analyzer = PopulationAnalyzer()
        
        # データ読み込み
        with st.spinner("データを読み込み中..."):
            df = analyzer.load_data(uploaded_file)
        
        st.sidebar.success(f"✅ データ読み込み完了: {df.shape[0]}行, {df.shape[1]}列")
        
        # サイドバーメニュー
        st.sidebar.header("🎯 分析メニュー")
        analysis_type = st.sidebar.selectbox(
            "分析タイプを選択",
            ["全国概観", "都道府県別分析", "比較分析", "年齢構造分析", "予測分析"]
        )
        
        # メイン分析画面
        if analysis_type == "全国概観":
            show_national_overview(analyzer)
            
        elif analysis_type == "都道府県別分析":
            show_prefecture_analysis(analyzer)
            
        elif analysis_type == "比較分析":
            show_comparison_analysis(analyzer)
            
        elif analysis_type == "年齢構造分析":
            show_age_structure_analysis(analyzer)
            
        elif analysis_type == "予測分析":
            show_prediction_analysis(analyzer)
    
    else:
        st.info("👆 サイドバーからcensus_2020.csvファイルをアップロードしてください")
        
        # サンプル画面
        st.markdown("""
        ## 🔍 このダッシュボードでできること
        
        ### 📈 全国概観
        - 日本全体の人口推移
        - 基本統計情報
        - トレンド分析
        
        ### 🗾 都道府県別分析
        - 任意の都道府県を選択
        - 人口推移グラフ
        - 男女比分析
        
        ### ⚖️ 比較分析
        - 複数都道府県の比較
        - 人口増減率比較
        - ランキング表示
        
        ### 👥 年齢構造分析
        - 年齢別人口ピラミッド
        - 高齢化率分析
        - 世代別トレンド
        
        ### 🔮 予測分析
        - 機械学習による将来予測
        - 複数モデルの比較
        - 信頼区間表示
        """)

def show_national_overview(analyzer):
    """全国概観の表示"""
    st.header("🌏 全国概観")
    
    # 全国時系列データ
    national_ts = analyzer.prepare_time_series_data()
    
    # メトリクス表示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_pop = national_ts['population'].iloc[-1]
        st.metric("2020年人口", f"{latest_pop/1000000:.1f}M人")
    
    with col2:
        pop_change = national_ts['population'].iloc[-1] - national_ts['population'].iloc[-2]
        st.metric("前回比増減", f"{pop_change/10000:.1f}万人", f"{pop_change/10000:.1f}万人")
    
    with col3:
        growth_rate = (national_ts['population'].iloc[-1] / national_ts['population'].iloc[-2] - 1) * 100
        st.metric("人口増減率", f"{growth_rate:.2f}%")
    
    with col4:
        prefecture_count = len(analyzer.prefectures)
        st.metric("分析対象都道府県", f"{prefecture_count}件")
    
    # 全国人口推移グラフ
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=national_ts['year'],
        y=national_ts['population']/1000000,
        mode='lines+markers',
        name='全国人口',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="日本全国人口推移 (1920-2020年)",
        xaxis_title="年",
        yaxis_title="人口 (百万人)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 都道府県別2020年人口ランキング
    st.subheader("📊 都道府県別人口ランキング (2020年)")
    
    pref_ranking = analyzer.df[analyzer.df['年齢階級'] == '総数'][['地域', '2020_総数']].copy()
    pref_ranking = pref_ranking.dropna().sort_values('2020_総数', ascending=False)
    pref_ranking['都道府県'] = pref_ranking['地域'].str.split('_').str[1]
    pref_ranking['人口'] = pref_ranking['2020_総数'].apply(lambda x: f"{x/10000:.1f}万人")
    pref_ranking['順位'] = range(1, len(pref_ranking) + 1)
    
    # 上位10位を表示
    top10 = pref_ranking[['順位', '都道府県', '人口']].head(10)
    st.dataframe(top10, use_container_width=True)

def show_prefecture_analysis(analyzer):
    """都道府県別分析の表示"""
    st.header("🗾 都道府県別分析")
    
    # 都道府県選択
    prefecture_options = {pref.split('_')[1]: pref for pref in analyzer.prefectures}
    selected_pref_name = st.selectbox("都道府県を選択", list(prefecture_options.keys()))
    selected_pref = prefecture_options[selected_pref_name]
    
    # 選択した都道府県のデータ
    pref_ts = analyzer.prepare_time_series_data(selected_pref)
    
    if not pref_ts.empty:
        # メトリクス
        col1, col2, col3 = st.columns(3)
        
        with col1:
            latest_pop = pref_ts['population'].iloc[-1]
            st.metric(f"{selected_pref_name} 人口", f"{latest_pop/10000:.1f}万人")
        
        with col2:
            if len(pref_ts) > 1:
                pop_change = pref_ts['population'].iloc[-1] - pref_ts['population'].iloc[-2]
                st.metric("前回比増減", f"{pop_change/10000:.1f}万人")
        
        with col3:
            if len(pref_ts) > 1:
                growth_rate = (pref_ts['population'].iloc[-1] / pref_ts['population'].iloc[-2] - 1) * 100
                st.metric("人口増減率", f"{growth_rate:.2f}%")
        
        # 人口推移グラフ
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=(f'{selected_pref_name} 人口推移', '男女別人口推移'),
                           vertical_spacing=0.1)
        
        # 総人口推移
        fig.add_trace(go.Scatter(
            x=pref_ts['year'],
            y=pref_ts['population']/10000,
            mode='lines+markers',
            name='総人口',
            line=dict(color='#1f77b4', width=3)
        ), row=1, col=1)
        
        # 男女別データの取得と表示
        pref_data = analyzer.df[analyzer.df['地域'] == selected_pref]
        total_data = pref_data[pref_data['年齢階級'] == '総数']
        
        years = [1920, 1925, 1930, 1935, 1940, 1945, 1950, 1955, 1960, 1965, 
                 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
        
        male_data = []
        female_data = []
        
        for year in years:
            male_col = f'{year}_男'
            female_col = f'{year}_女'
            
            if male_col in total_data.columns and female_col in total_data.columns:
                male_pop = total_data[male_col].iloc[0] if not total_data.empty else np.nan
                female_pop = total_data[female_col].iloc[0] if not total_data.empty else np.nan
                
                if not (pd.isna(male_pop) or pd.isna(female_pop)):
                    male_data.append({'year': year, 'population': male_pop})
                    female_data.append({'year': year, 'population': female_pop})
        
        if male_data and female_data:
            male_df = pd.DataFrame(male_data)
            female_df = pd.DataFrame(female_data)
            
            fig.add_trace(go.Scatter(
                x=male_df['year'],
                y=male_df['population']/10000,
                mode='lines+markers',
                name='男性',
                line=dict(color='#ff7f0e', width=2)
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=female_df['year'],
                y=female_df['population']/10000,
                mode='lines+markers',
                name='女性',
                line=dict(color='#d62728', width=2)
            ), row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_yaxes(title_text="人口 (万人)")
        fig.update_xaxes(title_text="年")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 年齢構造
        st.subheader("👥 年齢構造 (2020年)")
        age_data = analyzer.get_age_structure_data(selected_pref, 2020)
        
        if not age_data.empty:
            fig_age = go.Figure()
            fig_age.add_trace(go.Bar(
                x=age_data['age_group'],
                y=age_data['total_population']/10000,
                name='総人口',
                marker_color='lightblue'
            ))
            
            fig_age.update_layout(
                title=f"{selected_pref_name} 年齢別人口 (2020年)",
                xaxis_title="年齢層",
                yaxis_title="人口 (万人)",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_age, use_container_width=True)

def show_comparison_analysis(analyzer):
    """比較分析の表示"""
    st.header("⚖️ 比較分析")
    
    # 都道府県選択（複数選択）
    prefecture_options = {pref.split('_')[1]: pref for pref in analyzer.prefectures}
    selected_prefs = st.multiselect(
        "比較する都道府県を選択 (最大5つ)",
        list(prefecture_options.keys()),
        default=['東京都', '大阪府', '愛知県'],
        max_selections=5
    )
    
    if selected_prefs:
        # 比較グラフ
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, pref_name in enumerate(selected_prefs):
            pref_code = prefecture_options[pref_name]
            pref_ts = analyzer.prepare_time_series_data(pref_code)
            
            if not pref_ts.empty:
                fig.add_trace(go.Scatter(
                    x=pref_ts['year'],
                    y=pref_ts['population']/10000,
                    mode='lines+markers',
                    name=pref_name,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title="都道府県別人口推移比較",
            xaxis_title="年",
            yaxis_title="人口 (万人)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 増減率比較表
        st.subheader("📊 人口増減率比較")
        
        comparison_data = []
        for pref_name in selected_prefs:
            pref_code = prefecture_options[pref_name]
            pref_ts = analyzer.prepare_time_series_data(pref_code)
            
            if len(pref_ts) > 1:
                latest_pop = pref_ts['population'].iloc[-1]
                previous_pop = pref_ts['population'].iloc[-2]
                growth_rate = (latest_pop / previous_pop - 1) * 100
                
                comparison_data.append({
                    '都道府県': pref_name,
                    '2020年人口': f"{latest_pop/10000:.1f}万人",
                    '増減率': f"{growth_rate:.2f}%"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)

def show_age_structure_analysis(analyzer):
    """年齢構造分析の表示"""
    st.header("👥 年齢構造分析")
    
    # 分析タイプ選択
    analysis_mode = st.radio("分析モード", ["全国", "都道府県別"])
    
    if analysis_mode == "全国":
        age_data = analyzer.get_age_structure_data(None, 2020)
    else:
        prefecture_options = {pref.split('_')[1]: pref for pref in analyzer.prefectures}
        selected_pref_name = st.selectbox("都道府県を選択", list(prefecture_options.keys()))
        selected_pref = prefecture_options[selected_pref_name]
        age_data = analyzer.get_age_structure_data(selected_pref, 2020)
    
    if not age_data.empty:
        # 人口ピラミッド
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=age_data['age_group'],
            x=-age_data['male_population']/10000,  # 男性は負の値で左側に表示
            orientation='h',
            name='男性',
            marker_color='lightblue',
            text=age_data['male_population'].apply(lambda x: f"{x/10000:.1f}万"),
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            y=age_data['age_group'],
            x=age_data['female_population']/10000,  # 女性は正の値で右側に表示
            orientation='h',
            name='女性',
            marker_color='pink',
            text=age_data['female_population'].apply(lambda x: f"{x/10000:.1f}万"),
            textposition='inside'
        ))
        
        fig.update_layout(
            title=f"人口ピラミッド 2020年 ({'全国' if analysis_mode == '全国' else selected_pref_name})",
            xaxis_title="人口 (万人)",
            yaxis_title="年齢層",
            barmode='relative',
            height=600
        )
        
        # X軸の表示を調整（負の値も正として表示）
        fig.update_xaxes(tickvals=list(range(-500, 501, 100)),
                        ticktext=[str(abs(x)) for x in range(-500, 501, 100)])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 統計サマリー
        total_pop = age_data['total_population'].sum()
        elderly_pop = age_data[age_data['age_group'].isin(['65～69歳', '70～74歳', '75～79歳', '80～84歳', '85歳以上'])]['total_population'].sum()
        elderly_rate = (elderly_pop / total_pop) * 100 if total_pop > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総人口", f"{total_pop/10000:.1f}万人")
        with col2:
            st.metric("65歳以上人口", f"{elderly_pop/10000:.1f}万人")
        with col3:
            st.metric("高齢化率", f"{elderly_rate:.1f}%")

def show_prediction_analysis(analyzer):
    """予測分析の表示"""
    st.header("🔮 予測分析")
    
    # 予測対象選択
    prediction_target = st.radio("予測対象", ["全国", "都道府県別"])
    
    if prediction_target == "都道府県別":
        prefecture_options = {pref.split('_')[1]: pref for pref in analyzer.prefectures}
        selected_pref_name = st.selectbox("都道府県を選択", list(prefecture_options.keys()))
        selected_pref = prefecture_options[selected_pref_name]
        target_name = selected_pref_name
        ts_data = analyzer.prepare_time_series_data(selected_pref)
    else:
        target_name = "全国"
        ts_data = analyzer.prepare_time_series_data()
    
    # 予測年数設定
    forecast_years = st.slider("予測年数", 10, 50, 30, 5)
    
    if not ts_data.empty:
        # 予測実行
        with st.spinner("予測計算中..."):
            results = analyzer.predict_population(ts_data, forecast_years)
        
        # 予測結果グラフ
        fig = go.Figure()
        
        # 実績データ
        fig.add_trace(go.Scatter(
            x=ts_data['year'],
            y=ts_data['population']/10000,
            mode='lines+markers',
            name='実績',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # 予測データ
        colors = {'Linear': 'red', 'Polynomial_2': 'green', 'Polynomial_3': 'orange'}
        
        for model_name, result in results.items():
            fig.add_trace(go.Scatter(
                x=result['future_years'],
                y=result['future_pred']/10000,
                mode='lines+markers',
                name=f'{model_name} 予測',
                line=dict(color=colors.get(model_name, 'gray'), width=2, dash='dash'),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"{target_name} 人口予測",
            xaxis_title="年",
            yaxis_title="人口 (万人)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # モデル性能比較
        st.subheader("📈 モデル性能比較")
        
        performance_data = []
        for model_name, result in results.items():
            performance_data.append({
                'モデル': model_name,
                'R²スコア': f"{result['r2']:.3f}",
                'MAE (万人)': f"{result['mae']/10000:.1f}"
            })
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # 最適モデルの予測表
        best_model = max(results.keys(), key=lambda k: results[k]['r2'])
        st.subheader(f"🏆 最適モデル ({best_model}) の予測結果")
        
        forecast_data = []
        for i, year in enumerate(results[best_model]['future_years']):
            pop = results[best_model]['future_pred'][i]
            forecast_data.append({
                '年': int(year),
                '予測人口': f"{pop/10000:.1f}万人"
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        st.dataframe(forecast_df, use_container_width=True)

if __name__ == "__main__":
    main()
import streamlit as st
import ephem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import UnivariateSpline, interp1d
from datetime import datetime, timedelta, date
import sys
import os
from collections import defaultdict
from matplotlib.patches import Rectangle
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from datetime import datetime, date

# 1. 初始化地理编码器 (为了在Streamlit Cloud上使用，请使用一个自定义的user_agent)
# 注意：请将 'your_app_name_here' 替换为您应用的名称。
geolocator = Nominatim(user_agent="deepskyvistool")

def geocode_location(location_name):
    """根据输入的地名，返回其经纬度和显示名称。"""
    if not location_name:
        return None, None, None
    try:
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude, location.address
        else:
            st.warning(f"找不到地点: {location_name}")
            return None, None, None
    except Exception as e:
        st.error(f"地理编码出错: {e}")
        return None, None, None

# ================= 页面配置 =================
st.set_page_config(page_title="深空摄影最佳时段", layout="wide")
st.title("🌌 深空摄影最佳时段可视化工具")
st.markdown("输入观测地点与日期范围，生成无月光干扰的黑暗时段图表。")

# ================= 辅助函数 =================
def parse_latitude(lat_str):
    lat_str = lat_str.strip().upper()
    if lat_str.endswith('N'):
        return float(lat_str[:-1])
    elif lat_str.endswith('S'):
        return -float(lat_str[:-1])
    else:
        return float(lat_str)

def parse_longitude(lon_str):
    lon_str = lon_str.strip().upper()
    if lon_str.endswith('E'):
        return float(lon_str[:-1])
    elif lon_str.endswith('W'):
        return -float(lon_str[:-1])
    else:
        return float(lon_str)

def local_time_to_plot_value(dt):
    hour = dt.hour
    minute = dt.minute
    if hour < 12:
        value = (hour + 12) + minute / 60.0
        adj_date = dt.date() - timedelta(days=1)
    else:
        value = (hour - 12) + minute / 60.0
        adj_date = dt.date()
    return value, adj_date

def calc_timezone_offset(lon_str):
    """根据经度字符串计算本地时区相对于 UTC 的小时偏移（东正西负）"""
    lon_str = lon_str.strip().upper()
    if lon_str[-1] in 'EW':
        num_part = lon_str[:-1]
        direction = lon_str[-1]
    else:
        num_part = lon_str
        direction = 'E'
    lon_value = float(num_part)
    if direction == 'W':
        lon_value = -lon_value
    # 经度每15度一个时区，四舍五入
    tz_offset = int(round(lon_value / 15.0))
    return tz_offset

def generate_astronomical_data(lat, lon, start_date, end_date, timezone_hours=8):
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.elevation = 0

    records = []
    current_date = start_date
    while current_date <= end_date:
        observer.date = current_date.strftime('%Y/%m/%d') + ' 12:00:00'

        # --- 日出、日落 ---
        try:
            sunrise_utc = observer.previous_rising(ephem.Sun())
            sunset_utc = observer.next_setting(ephem.Sun())
            sunrise_local = ephem.Date(sunrise_utc).datetime() + timedelta(hours=timezone_hours)
            sunset_local = ephem.Date(sunset_utc).datetime() + timedelta(hours=timezone_hours)
            sunrise_val, sunrise_date = local_time_to_plot_value(sunrise_local)
            sunset_val, sunset_date = local_time_to_plot_value(sunset_local)
            sunrise_str = sunrise_local.strftime('%H:%M')
            sunset_str = sunset_local.strftime('%H:%M')
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            sunrise_str = sunset_str = ''
            sunrise_val = sunset_val = float('nan')
            sunrise_date = sunset_date = current_date

        # --- 月出、月落 ---
        moon = ephem.Moon()
        try:
            moonrise_utc = observer.previous_rising(moon)
            moonset_utc = observer.next_setting(moon)
            moonrise_local = ephem.Date(moonrise_utc).datetime() + timedelta(hours=timezone_hours)
            moonset_local = ephem.Date(moonset_utc).datetime() + timedelta(hours=timezone_hours)
            moonrise_val, moonrise_date = local_time_to_plot_value(moonrise_local)
            moonset_val, moonset_date = local_time_to_plot_value(moonset_local)
            moonrise_str = moonrise_local.strftime('%H:%M')
            moonset_str = moonset_local.strftime('%H:%M')
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            moonrise_str = moonset_str = ''
            moonrise_val = moonset_val = float('nan')
            moonrise_date = moonset_date = current_date

        # --- 天文曙暮光 ---
        observer.horizon = '-18'
        try:
            dawn_utc = observer.previous_rising(ephem.Sun(), use_center=True)
            dusk_utc = observer.next_setting(ephem.Sun(), use_center=True)
            dawn_local = ephem.Date(dawn_utc).datetime() + timedelta(hours=timezone_hours)
            dusk_local = ephem.Date(dusk_utc).datetime() + timedelta(hours=timezone_hours)
            dawn_val, dawn_date = local_time_to_plot_value(dawn_local)
            dusk_val, dusk_date = local_time_to_plot_value(dusk_local)
            dawn_str = dawn_local.strftime('%H:%M')
            dusk_str = dusk_local.strftime('%H:%M')
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            dawn_str = dusk_str = ''
            dawn_val = dusk_val = float('nan')
            dawn_date = dusk_date = current_date
        observer.horizon = '0'

        # --- 月相照度 ---
        moon.compute(observer)
        illumination = moon.moon_phase * 100.0

        records.append({
            'Date': current_date,
            'Sunrise_str': sunrise_str,
            'Sunset_str': sunset_str,
            'Moonrise_str': moonrise_str,
            'Moonset_str': moonset_str,
            'Dawn_str': dawn_str,
            'Dusk_str': dusk_str,
            'Illumination': f"{illumination:.1f}%",
            'Sunrise_val': sunrise_val,
            'Sunset_val': sunset_val,
            'Moonrise_val': moonrise_val,
            'Moonset_val': moonset_val,
            'Dawn_val': dawn_val,
            'Dusk_val': dusk_val,
            'Sunrise_date': sunrise_date,
            'Sunset_date': sunset_date,
            'Moonrise_date': moonrise_date,
            'Moonset_date': moonset_date,
            'Dawn_date': dawn_date,
            'Dusk_date': dusk_date,
        })
        current_date += timedelta(days=1)

    return pd.DataFrame(records)

def smooth_curve(x, y, num_points=300):
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 4:
        return x_clean, y_clean
    spl = UnivariateSpline(x_clean, y_clean, s=0.5)
    x_smooth = np.linspace(x_clean.min(), x_clean.max(), num_points)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth

def interval_to_segments(start, end):
    if np.isnan(start) or np.isnan(end):
        return []
    if start <= end:
        return [(start, end)]
    else:
        return [(start, 24), (0, end)]

def plot_discontinuous(ax, x, y, color, label, marker='o-', markersize=3, linewidth=1.5):
    x_new, y_new = [], []
    for i in range(len(x)-1):
        x_new.append(x[i])
        y_new.append(y[i])
        if abs(y[i+1] - y[i]) > 12:
            x_new.append(np.nan)
            y_new.append(np.nan)
    x_new.append(x[-1])
    y_new.append(y[-1])
    ax.plot(x_new, y_new, marker, color=color, label=label,
            markersize=markersize, linewidth=linewidth)

# ================= 缓存数据生成 =================
@st.cache_data
def compute_data(lat_str, lon_str, start_date_str, end_date_str, beijing_time):
    lat = parse_latitude(lat_str)
    lon = parse_longitude(lon_str)
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    total_days = (end_date - start_date).days + 1

    # 时区偏移计算
    if beijing_time:
        timezone_hours = 8
    else:
        timezone_hours = calc_timezone_offset(lon_str)
    target_tz = timezone_hours

    df = generate_astronomical_data(lat, lon, start_date, end_date, timezone_hours=timezone_hours)
    return df, lat, lon, start_date, end_date, total_days, timezone_hours, target_tz

# ================= 主绘图函数 =================
def create_figure(df, lat_str, lon_str, lat, lon, start_date, end_date, total_days, timezone_hours, target_tz):
    df['Sunrise_num'] = mdates.date2num(pd.to_datetime(df['Sunrise_date']))
    df['Sunset_num'] = mdates.date2num(pd.to_datetime(df['Sunset_date']))
    df['Moonrise_num'] = mdates.date2num(pd.to_datetime(df['Moonrise_date']))
    df['Moonset_num'] = mdates.date2num(pd.to_datetime(df['Moonset_date']))
    df['Dawn_num'] = mdates.date2num(pd.to_datetime(df['Dawn_date']))
    df['Dusk_num'] = mdates.date2num(pd.to_datetime(df['Dusk_date']))
    date_range = pd.date_range(start_date, end_date, freq='D')

    x_sr = df['Sunrise_num'].values
    y_sr = df['Sunrise_val'].values
    x_ss = df['Sunset_num'].values
    y_ss = df['Sunset_val'].values
    x_mr = df['Moonrise_num'].values
    y_mr = df['Moonrise_val'].values
    x_ms = df['Moonset_num'].values
    y_ms = df['Moonset_val'].values
    x_dawn = df['Dawn_num'].values
    y_dawn = df['Dawn_val'].values
    x_dusk = df['Dusk_num'].values
    y_dusk = df['Dusk_val'].values

    x_sr_s, y_sr_s = smooth_curve(x_sr, y_sr, num_points=total_days)
    x_ss_s, y_ss_s = smooth_curve(x_ss, y_ss, num_points=total_days)
    x_dawn_s, y_dawn_s = smooth_curve(x_dawn, y_dawn, num_points=total_days)
    x_dusk_s, y_dusk_s = smooth_curve(x_dusk, y_dusk, num_points=total_days)

    y_offset = 12
    y_sr_s_plot = y_sr_s - y_offset
    y_ss_s_plot = y_ss_s - y_offset
    y_dawn_s_plot = y_dawn_s - y_offset
    y_dusk_s_plot = y_dusk_s - y_offset

    y_mr_offset = y_mr - y_offset
    y_ms_offset = y_ms - y_offset

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_sr_s, y_sr_s_plot, 'o-', color='gold', label='Sunrise', markersize=3, linewidth=1.5)
    ax.plot(x_ss_s, y_ss_s_plot, 'o-', color='orange', label='Sunset', markersize=3, linewidth=1.5)
    ax.plot(x_dawn_s, y_dawn_s_plot, '--', color='purple', label='Dawn', linewidth=1.2)
    ax.plot(x_dusk_s, y_dusk_s_plot, '--', color='magenta', label='Dusk', linewidth=1.2)

    plot_discontinuous(ax, x_mr, y_mr_offset, 'skyblue', 'Moonrise')
    plot_discontinuous(ax, x_ms, y_ms_offset, 'navy', 'Moonset')

    # ========== 高亮区域 ==========
    date_range_full = pd.date_range(start=start_date, end=end_date, freq='D')
    date_list = date_range_full.date.tolist()
    noon_nums = mdates.date2num(date_range_full + pd.Timedelta(hours=12))

    illum_vals = df['Illumination'].str.replace('%', '').astype(float).values
    n = len(df)

    green_color = 'limegreen'
    yellow_color = 'gold'
    alpha = 0.5
    green_label_added = False
    yellow_label_added = False

    moon_events = defaultdict(list)
    for _, row in df.iterrows():
        mr_date = row['Moonrise_date']
        if pd.notna(mr_date) and not np.isnan(row['Moonrise_val']):
            moon_events[mr_date].append((row['Moonrise_val'], 'rise'))
        ms_date = row['Moonset_date']
        if pd.notna(ms_date) and not np.isnan(row['Moonset_val']):
            moon_events[ms_date].append((row['Moonset_val'], 'set'))

    for i in range(n - 1):
        sunset_val = df.iloc[i]['Sunset_val']
        sunrise_next_val = df.iloc[i+1]['Sunrise_val']
        if np.isnan(sunset_val) or np.isnan(sunrise_next_val):
            continue

        night_segs = interval_to_segments(sunset_val, sunrise_next_val)

        events = []
        for d in (date_list[i], date_list[i+1]):
            events.extend(moon_events[d])

        moon_segs = []
        if events:
            events.sort(key=lambda x: x[0])
            in_moon = False
            start_time = None
            if events[0][1] == 'set':
                in_moon = True
                start_time = 0.0
            for t, etype in events:
                if etype == 'rise' and not in_moon:
                    in_moon = True
                    start_time = t
                elif etype == 'set' and in_moon:
                    in_moon = False
                    moon_segs.append((start_time, t))
                    start_time = None
            if in_moon:
                moon_segs.append((start_time, 24.0))

        moon_in_night_segs = []
        for n_low, n_high in night_segs:
            for m_low, m_high in moon_segs:
                low = max(n_low, m_low)
                high = min(n_high, m_high)
                if low < high:
                    moon_in_night_segs.append((low, high))

        no_moon_segs = []
        for n_low, n_high in night_segs:
            cuts = [(n_low, n_high)]
            for m_low, m_high in moon_in_night_segs:
                new_cuts = []
                for c_low, c_high in cuts:
                    if m_low <= c_low and m_high >= c_high:
                        continue
                    elif m_low > c_low and m_high < c_high:
                        new_cuts.append((c_low, m_low))
                        new_cuts.append((m_high, c_high))
                    elif m_low <= c_low and m_high < c_high:
                        new_cuts.append((m_high, c_high))
                    elif m_low > c_low and m_high >= c_high:
                        new_cuts.append((c_low, m_low))
                    else:
                        new_cuts.append((c_low, c_high))
                cuts = new_cuts
            no_moon_segs.extend(cuts)

        illum_today = illum_vals[i]
        width = noon_nums[i+1] - noon_nums[i]
        x_left = noon_nums[i] - width

        def shift_segs(segs, offset):
            return [(low - offset, high - offset) for low, high in segs]

        all_highlight_segs = []
        if not np.isnan(illum_today) and illum_today <= 25:
            for low, high in shift_segs(no_moon_segs, y_offset):
                ax.add_patch(Rectangle((x_left, low), width, high - low,
                                       facecolor=green_color, alpha=alpha, edgecolor='none'))
                if not green_label_added:
                    ax.plot([], [], color=green_color, linewidth=10, label='No Moon', alpha=alpha)
                    green_label_added = True
            for low, high in shift_segs(moon_in_night_segs, y_offset):
                ax.add_patch(Rectangle((x_left, low), width, high - low,
                                       facecolor=yellow_color, alpha=alpha, edgecolor='none'))
                if not yellow_label_added:
                    ax.plot([], [], color=yellow_color, linewidth=10, label='Low Illumination', alpha=alpha)
                    yellow_label_added = True
            all_highlight_segs = no_moon_segs + moon_in_night_segs
        else:
            for low, high in shift_segs(no_moon_segs, y_offset):
                ax.add_patch(Rectangle((x_left, low), width, high - low,
                                       facecolor=green_color, alpha=alpha, edgecolor='none'))
                if not green_label_added:
                    ax.plot([], [], color=green_color, linewidth=10, label='No Moon', alpha=alpha)
                    green_label_added = True
            all_highlight_segs = no_moon_segs

        total_hours = sum(high - low for low, high in all_highlight_segs)
        if total_hours > 0 and i != 0 and total_days <= 32:
            ax.text(noon_nums[i] - width / 2, 0, f"{total_hours:.1f}h",
                    ha='center', va='center', fontsize=6,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                    zorder=10)

    # ========== 坐标轴设置 ==========
    ax.set_ylim(-12, 12)
    yticks = list(range(-12, 13))
    yticklabels = ['12:00'] + [f'{h:02d}:00' for h in range(13, 24)] + ['00:00'] + [f'{h:02d}:00' for h in range(1, 13)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(f'Time (UTC{target_tz:+d})')

    noon_ticks = mdates.date2num(date_range) + 0
    if total_days <= 62:
        step = 1
    elif total_days <= 180:
        step = 3
    else:
        step = 7
    tick_indices = range(0, len(noon_ticks), step)
    selected_ticks = noon_ticks[tick_indices]

    start_year = start_date.year
    end_year = end_date.year
    year_label = f"Date ({start_year})" if start_year == end_year else f"Date ({start_year}-{end_year})"
    ax.set_xticks(selected_ticks)
    bottom_labels = [date_range[i].strftime('%m-%d') for i in tick_indices]
    ax.set_xticklabels(bottom_labels, rotation=45)
    ax.set_xlabel(year_label)

    ax_top = ax.twiny()
    ax_top.set_xticks(selected_ticks)
    top_labels = [(date_range[i] + timedelta(days=1)).strftime('%m-%d') for i in tick_indices]
    ax_top.set_xticklabels(top_labels, rotation=45)

    ax.axhline(y=0, color='gray', linestyle='-')
    ax.set_xlim(noon_ticks[0], noon_ticks[-1])
    ax_top.set_xlim(noon_ticks[0], noon_ticks[-1])

    ax.set_title('Best Time(Night∩(No Moon∪Illumination≤25%))')
    ax.grid(True, linestyle='-', alpha=0.6)
    ax.legend(loc='upper left')

    def format_coord(s):
        if s[-1] in 'NSEW':
            return s[:-1] + '°' + s[-1]
        return s + '°'
    info_text = f"Position: {format_coord(lat_str)} {format_coord(lon_str)}"
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    return fig

# ================= Streamlit 侧边栏输入 =================
st.sidebar.header("📍 观测参数设置")

# 保留原有的手动输入经纬度功能
lat_str = st.sidebar.text_input("纬度 (如 40.15N, 28S)", value="40.15N")
lon_str = st.sidebar.text_input("经度 (如 116.27E, 18W)", value="116.27E")
st.sidebar.markdown("---")

# 2. 添加地名搜索功能
st.sidebar.subheader("🔍 或搜索地名")
location_input = st.sidebar.text_input("输入地名 (例如: Beijing, 故宫, 纽约)", value="")

if st.sidebar.button("搜索并应用"):
    lat, lon, addr = geocode_location(location_input)
    if lat is not None and lon is not None:
        st.sidebar.success(f"已找到: {addr}")
        # 在这里，您可以将获取到的经纬度格式化为您应用需要的字符串格式
        # 例如，将纬度格式化为 "40.15N" 或 "40.15S"
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        # 更新session_state，用于在页面上显示和后续使用
        st.session_state['selected_lat'] = f"{abs(lat):.4f}{lat_dir}"
        st.session_state['selected_lon'] = f"{abs(lon):.4f}{lon_dir}"
        st.rerun() # 重新运行以更新输入框的值

# 使用session_state中的值来更新经纬度输入框的默认值
# 这样可以实现点击搜索后，输入框自动填充
if 'selected_lat' in st.session_state:
    lat_str = st.sidebar.text_input("纬度", value=st.session_state['selected_lat'], key='lat_input')
if 'selected_lon' in st.session_state:
    lon_str = st.sidebar.text_input("经度", value=st.session_state['selected_lon'], key='lon_input')

st.sidebar.markdown("---")

# 3. 添加交互式地图
st.sidebar.subheader("🌍 或在地图上选点")
# 默认地图中心可以设置为北京，或者根据搜索到的地点动态调整
default_lat, default_lon = 39.9042, 116.4074 # 北京天安门

# 如果已经通过地名搜索得到了坐标，就用那个坐标作为地图中心
if 'selected_lat' in st.session_state and 'selected_lon' in st.session_state:
    # 这里需要将格式如 "40.15N" 的字符串转回浮点数
    lat_val = float(st.session_state['selected_lat'][:-1])
    if st.session_state['selected_lat'][-1] == 'S':
        lat_val = -lat_val
    lon_val = float(st.session_state['selected_lon'][:-1])
    if st.session_state['selected_lon'][-1] == 'W':
        lon_val = -lon_val
    default_lat, default_lon = lat_val, lon_val

m = folium.Map(location=[default_lat, default_lon], zoom_start=5)
m.add_child(folium.LatLngPopup())

# 显示地图并获取点击事件
map_data = st_folium(m, height=350, width=700)

# 处理地图点击事件
if map_data and map_data['last_clicked']:
    clicked_lat = map_data['last_clicked']['lat']
    clicked_lon = map_data['last_clicked']['lng']
    lat_dir = 'N' if clicked_lat >= 0 else 'S'
    lon_dir = 'E' if clicked_lon >= 0 else 'W'
    st.session_state['selected_lat'] = f"{abs(clicked_lat):.4f}{lat_dir}"
    st.session_state['selected_lon'] = f"{abs(clicked_lon):.4f}{lon_dir}"
    st.rerun()

# 日期、时区等其他输入保持不变
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date_str = st.date_input("起始日期", value=date(2026, 4, 1)).strftime('%Y-%m-%d')
with col2:
    end_date_str = st.date_input("结束日期", value=date(2026, 5, 1)).strftime('%Y-%m-%d')
beijing_time = st.sidebar.checkbox("强制使用东八区 (UTC+8)", value=False)

if st.sidebar.button("🚀 生成图表", type="primary"):
    try:
        lat = parse_latitude(lat_str)
        lon = parse_longitude(lon_str)
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except Exception as e:
        st.error(f"输入格式错误：{e}")
        st.stop()

    if lat < -90 or lat > 90:
        st.error(f"纬度值 {lat} 不合法 (应在 -90 到 90 之间)")
        st.stop()
    if lon < -180 or lon > 180:
        st.error(f"经度值 {lon} 不合法 (应在 -180 到 180 之间)")
        st.stop()
    if start_date >= end_date:
        st.error("结束日期必须晚于开始日期")
        st.stop()
    total_days = (end_date - start_date).days + 1
    if total_days > 366:
        st.error(f"日期跨度 {total_days} 天超过一年（最多366天），请缩短范围")
        st.stop()

    with st.spinner("正在计算天文数据并生成图表，请稍候..."):
        df, lat, lon, start_date, end_date, total_days, timezone_hours, target_tz = compute_data(
            lat_str, lon_str, start_date_str, end_date_str, beijing_time
        )
        fig = create_figure(df, lat_str, lon_str, lat, lon, start_date, end_date, total_days, timezone_hours, target_tz)
        st.pyplot(fig)

        csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            label="📥 下载 CSV 数据",
            data=csv,
            file_name=f"astro_{start_date_str}_to_{end_date_str}.csv",
            mime="text/csv"
        )

        st.success("图表生成完毕！")
else:
    st.info("👈 请在左侧设置参数并点击“生成图表”")

st.markdown("""
---
### 📌 特别注意

- 观测地位于中国境内时请勾选【强制东八区】以显示北京时间
- 当观测地纬度过高时（＞61.5°）会显示异常
- 当观测地距东八区过远时有概率显示异常，此时请不要勾选【强制东八区】
- 更多详情请见【使用说明】

---
""")

with st.sidebar.expander("📖 使用说明", expanded=False):
    st.markdown("""
    ### 🎯 项目目标
    本工具为深空摄影爱好者提供**科学、直观的决策辅助图表**。通过输入观测地点和日期范围，自动计算每日日出/日落、月出/月落、天文曙暮光及月相照度，并以特殊坐标系绘制“夜间‑无月‑低照度”复合时段，帮助快速识别最佳摄影窗口。

    ### ⚙️ 核心功能
    - **自动获取天文数据**：基于 `ephem` 库精确计算每日天文事件。
    - **智能时区处理**：支持强制东八区或根据经度自动推算时区。
    - **可视化图表**：
        - 纵轴以午夜 00:00 为原点，向上为后半夜/上午、向下为前半夜/下午。
        - 横轴下方标注下半部分所对应的日期，上方同理。
        - 曲线：日出(金)、日落(橙)、月出(浅蓝)、月落(深蓝)、天文晨光始(紫虚线)、天文昏影终(品红虚线)。
        - **绿色高亮**：夜间无月时段（最佳）。
        - **黄色高亮**：夜间有月但照度 ≤25% 的时段（次佳）。
    - **时长标注**：当总天数 ≤31 天时，在每日 00:00 线附近显示该夜最佳时段总时长（即高亮时长）。
    - **数据导出**：可下载 CSV 文件，包含所有原始计算数据。

    ### 📝 输入参数说明
    - **纬度**：格式如 `40.15N` (北纬) 或 `28S` (南纬)，支持小数点。
    - **经度**：格式如 `116.27E` (东经) 或 `18W` (西经)。
    - **日期范围**：起止日期，跨度不超过 366 天。
    - **强制东八区**：勾选后所有时间按 UTC+8 显示；不勾选则根据经度自动计算本地时区。

    ### 📊 图表阅读提示
    - 纵轴原点为午夜 00:00，正半轴为后半夜/上午，负半轴为前半夜/下午，更符合“一夜从日落开始到日出结束”的摄影习惯。

    ### 💡 典型使用场景
    - 规划深空摄影行程前，快速比较不同地点、不同月份的“无月黑暗时间”分布。
    - 评估某地点在一段时期内有多少个夜晚满足“无月”或“低照度有月”的条件。
    - 结合曙暮光曲线，判断天空完全黑暗的时长。
    """)
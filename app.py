import streamlit as st
import pandas as pd
import joblib
from scipy.spatial import distance

# 加载模型
model_v = joblib.load('viscosity.pkl')
model_s = joblib.load('solids.pkl')

# 加载数据集
excel_data = pd.read_excel('data半(创造异常数据).xlsx')

# 固定的特征值
fixed_values = {
    '乳液A': 2066,
    '乳液F': 1240,
    '水溶液F': 250,
    '其它': 107.54
}


def calculate_proportional_values(total_value):
    fixed_total = 3971.54
    remaining_total = total_value / fixed_total
    proportional_values = {}
    for key, value in fixed_values.items():
        if key in ['乳液A', '乳液F', '水溶液F']:
            proportional_values[key] = round(value * remaining_total)
        else:
            proportional_values[key] = round(value * remaining_total, 2)
    return proportional_values


def find_closest_water_solution_e(input_features):
    feature_columns = ['乳液A', '乳液A粘度', '乳液A固含量',
                       '乳液F', '乳液F粘度', '乳液F固含量',
                       '水溶液E固含量', '水溶液F', '水溶液F固含量',
                       '其它', '其他固含量']
    feature_data = excel_data[feature_columns]
    distances = feature_data.apply(lambda row: distance.euclidean(row, input_features), axis=1)
    closest_index = distances.idxmin()
    closest_row = excel_data.loc[closest_index]
    return closest_row['水'], closest_row['水溶液E']


def adjust_values(user_input_values, expected_viscosity):
    water = user_input_values['水']
    solution_e = user_input_values['水溶液E']
    while True:
        if water < 0 or solution_e < 0:
            raise ValueError("水或水溶液E的值变成负数，程序停止运行。")
        if water > 100:
            raise ValueError("水的值超过100，程序停止运行。")
        if solution_e > 300:
            raise ValueError("水溶液E的值超过300，程序停止运行。")

        user_input = pd.DataFrame([user_input_values]).assign(水=water, 水溶液E=solution_e)
        selected_features_v = user_input[
            ['乳液A粘度', '乳液F粘度', '水溶液E', '水溶液F', '水', '乳液A固含量', '乳液F固含量']]
        selected_features_s = user_input[['乳液A固含量', '乳液F固含量', '水', '乳液A粘度', '水溶液E', '乳液F粘度']]

        predicted_viscosity = model_v.predict(selected_features_v)[0]
        predicted_solids = model_s.predict(selected_features_s)[0]
        viscosity_difference = predicted_viscosity - expected_viscosity

        if abs(viscosity_difference) <= 200:
            break

        if viscosity_difference > 0:
            water += 1
            solution_e -= 0.5
        else:
            water -= 1
            solution_e += 1

    total = (user_input_values['乳液A'] +
             user_input_values['乳液F'] +
             water +
             solution_e +
             user_input_values['水溶液F'] +
             user_input_values['其它'])

    relative_error = abs(viscosity_difference) / expected_viscosity * 100

    result = {
        "乳液A": user_input_values['乳液A'],
        "乳液F": user_input_values['乳液F'],
        "优化后的水量": water,
        "优化后的水溶液E量": solution_e,
        "水溶液F": user_input_values['水溶液F'],
        "其他": user_input_values['其它'],
        "预测黏度": predicted_viscosity,
        "预测固含量": predicted_solids,
        "总计": total,
        "黏度差": abs(viscosity_difference),
        "相对误差 (%)": relative_error,
    }

    return result


# Streamlit 界面代码
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""

st.set_page_config(page_title="粘度优化工具", layout="wide")
st.title("粘度优化工具")
st.markdown("""<style>div[data-testid="stSidebar"] {background-color: #f0f2f6;}</style>""", unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.header("🔧 输入参数")

# 用户输入的总量
user_input_total = st.sidebar.number_input("总量 (g)", value=4163.77)

# 乳液A和乳液F的黏度输入
乳液A粘度 = st.sidebar.number_input("乳液A粘度", value=3230)
乳液F粘度 = st.sidebar.number_input("乳液F粘度", value=4410)

# 其他固定参数
水溶液E固含量 = st.sidebar.number_input("水溶液E固含量", value=0.2)
水溶液F固含量 = st.sidebar.number_input("水溶液F固含量", value=0.2)
其它固含量 = st.sidebar.number_input("其他固含量", value=0.2)
水溶液F = st.sidebar.number_input("水溶液F", value=250)
其它 = st.sidebar.number_input("其它", value=112.24)
预期黏度 = st.sidebar.number_input("预期黏度", value=5000)

if st.sidebar.button("🚀 确认"):
    # 根据总量计算比例的乳液A、乳液F、水溶液F、其他
    proportional_values = calculate_proportional_values(user_input_total)

    # 输入的其他特征
    user_input_features = {
        '乳液A粘度': 乳液A粘度,
        '乳液F粘度': 乳液F粘度,
        '水溶液E固含量': 水溶液E固含量,
        '水溶液F固含量': 水溶液F固含量,
        '其他固含量': 其它固含量,
        '水溶液F': 水溶液F,
        '其它': 其它
    }

    user_input_features.update(proportional_values)

    # 预计的黏度
    expected_viscosity = 预期黏度

    # 根据输入特征查找最接近的水和水溶液E
    closest_water, closest_solution_e = find_closest_water_solution_e(
        [user_input_features[col] for col in user_input_features]
    )

    # 合并结果并调用 adjust_values
    user_input_values = {**user_input_features, '水': closest_water, '水溶液E': closest_solution_e}

    with st.spinner("⏳ 正在加载，请稍候..."):
        try:
            result = adjust_values(user_input_values, expected_viscosity)
            st.subheader("✨ 优化结果")
            st.success("优化成功！以下是结果：")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("优化后的水量", f"{result['优化后的水量']:.2f}")
                st.metric("预测黏度", f"{result['预测黏度']:.2f}")
                st.metric("黏度差", f"{result['黏度差']:.2f}")

            with col2:
                st.metric("优化后的水溶液E量", f"{result['优化后的水溶液E量']:.2f}")
                st.metric("预测固含量", f"{result['预测固含量'] * 100:.2f}%")
                st.metric("相对误差 (%)", f"{result['相对误差 (%)']:.2f}%")

            st.write(f"### 总计: **<span style='font-size:1.2em'>{result['总计']:.2f} g</span>**",
                     unsafe_allow_html=True)

        except ValueError as e:
            st.error(str(e))

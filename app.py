import streamlit as st
import pandas as pd
import joblib
from scipy.spatial import distance

# 加载模型
model_v = joblib.load('viscosity.pkl')
model_s = joblib.load('solids.pkl')

# 加载数据集
excel_data = pd.read_excel('data半(创造异常数据).xlsx')  # 假设数据保存在 data.xlsx 文件中



def calculate_proportional_values(fixed_total,total_value):
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

        selected_features_v = user_input[[
            '乳液A粘度', '乳液F粘度', '水溶液E', '水溶液F', '水', '乳液A固含量', '乳液F固含量']]
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

    total = (
            user_input_values['乳液A'] +
            user_input_values['乳液F'] +
            water +
            solution_e +
            user_input_values['水溶液F'] +
            user_input_values['其它']
    )

    relative_error = abs(viscosity_difference) / expected_viscosity * 100

    result = {
        "乳液A": user_input_values['乳液A'],
        "乳液F": user_input_values['乳液F'],
        "优化后的水量": water,
        "优化后的水溶液E量": solution_e,
        "水溶液F": user_input_values['水溶液F'],
        "其他": user_input_values['其它'],
        "预测黏度": predicted_viscosity,
        "预测固含量": predicted_solids*100,
        "总计": total,
        "黏度差": abs(viscosity_difference),
        "相对误差 (%)": relative_error,
    }

    return result


# Streamlit UI
hide_streamlit_style = """
    <style>
    /* 隐藏右上角 GitHub 图标 */
    #MainMenu {visibility: hidden;}
    /* 隐藏页脚 */
    footer {visibility: hidden;}
    /* 隐藏顶部的 Streamlit 菜单 */
    header {visibility: hidden;}
    </style>
"""
st.set_page_config(page_title="粘度优化工具", layout="wide")
# 页面标题
st.title("🧪 产品黏度优化工具")
st.markdown("""<style>div[data-testid="stSidebar"] {background-color: #f0f2f6;}</style>""", unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# 页面说明
st.markdown("""
本工具用于根据输入的原料和参数，优化产品的水和水溶液E量，从而达到预期的黏度。
""")

# 使用容器分块布局
input_columns = st.columns(2)

# 输入数据
with input_columns[0]:
    st.subheader("1. 输入配方")
    配方_乳液A = st.number_input("配方-乳液A", min_value=0, value=2066)
    配方_乳液F = st.number_input("配方-乳液F", min_value=0, value=1240)
    配方_水溶液E = st.number_input("配方-水溶液E", min_value=0, value=260)
    配方_水溶液F = st.number_input("配方-水溶液F", min_value=0, value=250)
    配方_水 = st.number_input("配方-水", min_value=0, value=48)
    配方_其他 = st.number_input("配方-其他", min_value=0.00, value=107.54)


with input_columns[1]:
    st.subheader("2. 输入材料数据")
    乳液A粘度 = st.number_input("乳液A粘度", min_value=0, value=3230)
    乳液A固含量 = st.number_input("乳液A固含量", min_value=0.000, max_value=1.000, value=0.556, step=0.001,format="%.3f")

    乳液F粘度 = st.number_input("乳液F粘度", min_value=0, value=4410)
    乳液F固含量 = st.number_input("乳液F固含量", min_value=0.000, max_value=1.0000, value=0.607, step=0.001,format="%.3f")
    水溶液E固含量 = st.number_input("水溶液E固含量", min_value=0.0, max_value=1.0, value=0.2, step=0.001)
    水溶液F固含量 = st.number_input("水溶液F固含量", min_value=0.0, max_value=1.0, value=0.2, step=0.001)
    其他固含量 = st.number_input("其他固含量", min_value=0.0, max_value=1.0, value=0.85, step=0.001)

# 预计的黏度
user_input_total = st.number_input("需求总计", min_value=0.0, max_value=10000.0, value=4163.77,
                                       step=0.01)
expected_viscosity = st.number_input("预计的黏度", min_value=0, value=5000)

# 按钮用于触发计算
if st.button("开始优化"):
    # 计算同比例的乳液A、乳液F、水溶液F、其他
    # 固定的特征值
    fixed_values = {
        '乳液A': 配方_乳液A,
        '乳液F': 配方_乳液F,
        '水溶液F': 配方_水溶液F,
        '其它': 配方_其他
    }
    fixed_total = 配方_乳液A+配方_乳液F+配方_水+配方_水溶液E+配方_水溶液F+配方_其他

    proportional_values = calculate_proportional_values(fixed_total,user_input_total)

    # 合并用户输入的其他特征
    user_input_features = {
        '乳液A粘度': 乳液A粘度, '乳液A固含量': 乳液A固含量,
        '乳液F粘度': 乳液F粘度, '乳液F固含量': 乳液F固含量,
        '水溶液E固含量': 水溶液E固含量, '水溶液F固含量': 水溶液F固含量,
        '其他固含量': 其他固含量
    }

    # 合并所有特征
    user_input_features.update(proportional_values)

    # 根据输入特征查找最接近的水和水溶液E
    closest_water, closest_solution_e = find_closest_water_solution_e(
        [user_input_features[col] for col in ['乳液A', '乳液A粘度', '乳液A固含量', '乳液F', '乳液F粘度', '乳液F固含量',
                                              '水溶液E固含量', '水溶液F', '水溶液F固含量', '其它', '其他固含量']])

    # 合并结果并调用 adjust_values
    user_input_values = {**user_input_features, '水': closest_water, '水溶液E': closest_solution_e}
    optimized_result = adjust_values(user_input_values, expected_viscosity)

    with st.spinner("⏳ 正在加载，请稍候..."):
        try:
            st.subheader("🔍 优化结果")
            col1, col2 = st.columns(2)
            optimized_result = adjust_values(user_input_values, expected_viscosity)

            st.markdown("""
                        <style>
                            .big-font {
                                font-size:20px !important;
                            }
                        </style>
                        """, unsafe_allow_html=True)

            with col1:
                st.markdown(f"<p class='big-font'>乳液A: {optimized_result['乳液A']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='big-font'>乳液F: {optimized_result['乳液F']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='big-font'>水溶液F: {optimized_result['水溶液F']}</p>",
                            unsafe_allow_html=True)
                st.markdown(f"<p class='big-font'>其他: {optimized_result['其他']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='big-font'>优化后的水量: {optimized_result['优化后的水量']}</p>",
                            unsafe_allow_html=True)
                st.markdown(f"<p class='big-font'>优化后的水溶液E量: {optimized_result['优化后的水溶液E量']}</p>",
                            unsafe_allow_html=True)

            with col2:
                st.markdown(f"<p class='big-font'>预测黏度: {optimized_result['预测黏度']}</p>",
                            unsafe_allow_html=True)
                st.markdown(f"<p class='big-font'>预测固含量: {optimized_result['预测固含量']:.2f} %</p>",
                            unsafe_allow_html=True)
                st.markdown(f"<p class='big-font'>总计: {optimized_result['总计']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='big-font'>黏度差: {optimized_result['黏度差']:.2f}</p>",
                            unsafe_allow_html=True)
                st.markdown(f"<p class='big-font'>相对误差 (%): {optimized_result['相对误差 (%)']:.2f}</p>",
                            unsafe_allow_html=True)

        except ValueError as e:
            st.error(str(e))
    # st.success("### 计算完成！您的优化配方已生成。")

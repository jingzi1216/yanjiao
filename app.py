import streamlit as st
import pandas as pd
import joblib

# 加载模型
model_v = joblib.load('viscosity.pkl')
model_s = joblib.load('solids.pkl')

def adjust_values(user_input_values, expected_viscosity):
    # 初始化水和水溶液E的值
    water = user_input_values['水']
    solution_e = user_input_values['水溶液E']

    while True:
        # 检查异常情况
        if water < 0 or solution_e < 0:
            raise ValueError("水或水溶液E的值变成负数，程序停止运行。")
        if water > 100:
            raise ValueError("水的值超过100，程序停止运行。")
        if solution_e > 300:
            raise ValueError("水溶液E的值超过300，程序停止运行。")

        # 更新用户输入
        user_input = pd.DataFrame([user_input_values]).assign(水=water, 水溶液E=solution_e)

        # 特征选择
        selected_features_v = user_input[['乳液A粘度', '乳液F粘度', '水溶液E', '水溶液F', '水', '乳液A固含量', '乳液F固含量']]
        selected_features_s = user_input[['乳液A固含量', '乳液F固含量', '水', '乳液A粘度', '水溶液E', '乳液F粘度']]

        # 预测黏度和固含量
        predicted_viscosity = model_v.predict(selected_features_v)[0]
        predicted_solids = model_s.predict(selected_features_s)[0]

        # 计算黏度误差
        viscosity_difference = predicted_viscosity - expected_viscosity

        # 判断是否在目标范围内
        if abs(viscosity_difference) <= 200:
            break

        # 根据预测黏度调整水和水溶液E的值
        if viscosity_difference > 0:  # 黏度过高
            water += 1
            solution_e -= 0.5
        else:  # 黏度过低
            water -= 1
            solution_e += 1

    # 计算总量
    total = (
        user_input_values['乳液A'] +
        user_input_values['乳液F'] +
        water +
        solution_e +
        user_input_values['水溶液F'] +
        user_input_values['其它']
    )

    # 计算相对误差
    relative_error = abs(viscosity_difference) / expected_viscosity * 100

    # 输出优化结果
    result = {
        "优化后的水量": water,
        "优化后的水溶液E量": solution_e,
        "预测黏度": predicted_viscosity,
        "预测固含量": predicted_solids,
        "总计": total,
        "黏度差": abs(viscosity_difference),
        "相对误差 (%)": relative_error,
    }

    return result

# Streamlit 界面
# Streamlit 界面
# 隐藏右上角的 GitHub 图标和其他 Streamlit 默认元素
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
st.title("粘度优化工具")
st.markdown("""<style>div[data-testid="stSidebar"] {background-color: #f0f2f6;}</style>""", unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.sidebar.header("🔧 输入参数")
乳液A = st.sidebar.number_input("乳液A ", value=2066)
乳液A粘度 = st.sidebar.number_input("乳液A粘度", value=3180)
乳液A固含量 = st.sidebar.number_input("乳液A固含量", value=0.5556,format="%.3f")
乳液F = st.sidebar.number_input("乳液F ", value=1240)
乳液F粘度 = st.sidebar.number_input("乳液F粘度", value=4740)
乳液F固含量 = st.sidebar.number_input("乳液F固含量", value=0.6030,format="%.3f")
水溶液E = st.sidebar.number_input("水溶液E ", value=210)
水溶液F = st.sidebar.number_input("水溶液F ", value=250)
水 = st.sidebar.number_input("水 ", value=75.6)
其它 = st.sidebar.number_input("其它 ", value=112.24)
预期黏度 = st.sidebar.number_input("预期黏度", value=5000)

if st.sidebar.button("🚀 确认"):
    user_input_values = {
        '乳液A': 乳液A, '乳液A粘度': 乳液A粘度, '乳液A固含量': 乳液A固含量,
        '乳液F': 乳液F, '乳液F粘度': 乳液F粘度, '乳液F固含量': 乳液F固含量,
        '水溶液E': 水溶液E, '水溶液F': 水溶液F, '水': 水, '其它': 其它
    }

    with st.spinner("⏳ 正在加载，请稍候..."):
        try:
            result = adjust_values(user_input_values, 预期黏度)
            st.subheader("✨ 优化结果")
            st.success("优化成功！以下是结果：")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("优化后的水量", f"{result['优化后的水量']:.2f} ")
                st.metric("预测黏度", f"{result['预测黏度']:.2f}")
                st.metric("黏度差", f"{result['黏度差']:.2f}")



            with col2:
                st.metric("优化后的水溶液E量", f"{result['优化后的水溶液E量']:.2f} ")
                st.metric("预测固含量", f"{result['预测固含量']*100:.2f}%")

                st.metric("相对误差 (%)", f"{result['相对误差 (%)']:.2f}%")

            st.write(f"### 总计: **<span style='font-size:1.2em'>{result['总计']:.2f} g</span>**", unsafe_allow_html=True)

        except ValueError as e:
            st.error(str(e))


import streamlit as st
import time
import pandas as pd
import joblib
from pyswarm import pso  # 使用 PySwarm 库中的 PSO 实现

# 加载模型
model_v = joblib.load('viscosity.pkl')
model_s = joblib.load('solids.pkl')

# 定义目标函数
def optimization_function(individual, *args):
    """
    PSO 的目标函数
    individual[0]: 水
    individual[1]: 水溶液E
    """
    expected_viscosity, user_input_values = args

    # 获取水和水溶液E的值
    water, solution_e = individual

    # 修改用户输入
    user_input = pd.DataFrame([user_input_values])
    user_input['水'] = water
    user_input['水溶液E'] = solution_e

    # 特征选择
    selected_features_v = user_input[['乳液A粘度', '乳液F粘度', '水溶液E', '水', '乳液A固含量', '乳液F固含量']]
    selected_features_s = user_input[['乳液A固含量', '乳液F固含量', '水', '乳液A粘度', '水溶液E', '乳液F粘度']]

    # 预测黏度和固含量
    predicted_viscosity = model_v.predict(selected_features_v)[0]
    predicted_solids = model_s.predict(selected_features_s)[0]

    # 计算惩罚项
    viscosity_penalty = max(0, 4500 - predicted_viscosity) + max(0, predicted_viscosity - 5500)
    viscosity_target_deviation = abs(predicted_viscosity - expected_viscosity)
    solids_penalty = max(0, 0.50 - predicted_solids) + max(0, predicted_solids - 0.54)

    # 确保总量约束满足
    input_total = (
        user_input_values['乳液A'] + user_input_values['乳液F'] + user_input_values['水'] +
        user_input_values['水溶液E'] + user_input_values['水溶液F'] + user_input_values['其它']
    )
    total = (
        user_input_values['乳液A'] + user_input_values['乳液F'] + water + solution_e +
        user_input_values['水溶液F'] + user_input_values['其它']
    )

    # 设定新的约束条件：总量在输入总量到输入总量+50之间
    total_lower_limit = input_total  # 输入总量下限
    total_upper_limit = input_total + 50  # 输入总量上限

    # 惩罚总量不在范围内的情况
    total_penalty = 0
    if total < total_lower_limit:  # 如果总量小于输入总量
        total_penalty = 10*(total_lower_limit - total)
    elif total > total_upper_limit:  # 如果总量大于输入总量+50
        total_penalty = 10*(total - total_upper_limit)

    total_closeness_penalty = 0
    if total >= input_total:
        total_closeness_penalty = abs(total - input_total)  # 总量偏离输入总量的惩罚
    else:
        total_closeness_penalty = (input_total - total)  # 总量小于输入总量的惩罚

    # 返回目标函数值
    return viscosity_penalty + solids_penalty + viscosity_target_deviation + total_penalty + total_closeness_penalty


# 粒子群算法求解
def run_pso(user_input_values, expected_viscosity):
    # 设置变量边界：水和水溶液E的范围
    lb = [10, 100]  # 下界：水和水溶液E
    ub = [100, 300]  # 上界：水和水溶液E

    # 运行 PSO 算法
    optimal_solution, fopt = pso(
        optimization_function,
        lb,
        ub,
        args=(expected_viscosity, user_input_values),
        swarmsize=50,  # 粒子群的大小
        maxiter=30     # 最大迭代次数
    )

    # 获取优化结果
    water, solution_e = optimal_solution

    # 特征选择
    selected_features_v = pd.DataFrame([user_input_values]).assign(水=water, 水溶液E=solution_e)[
        ['乳液A粘度', '乳液F粘度', '水溶液E', '水', '乳液A固含量', '乳液F固含量']]
    selected_features_s = pd.DataFrame([user_input_values]).assign(水=water, 水溶液E=solution_e)[
        ['乳液A固含量', '乳液F固含量', '水', '乳液A粘度', '水溶液E', '乳液F粘度']]

    # 预测黏度和固含量
    predicted_viscosity = model_v.predict(selected_features_v)[0]
    predicted_solids = model_s.predict(selected_features_s)[0]

    # 计算结果
    total = (
        user_input_values['乳液A'] +
        user_input_values['乳液F'] +
        water +
        solution_e +
        user_input_values['水溶液F'] +
        user_input_values['其它']
    )
    viscosity_difference = abs(predicted_viscosity - expected_viscosity)
    relative_error = viscosity_difference / expected_viscosity * 100

    # 输出优化结果
    result = {
        "优化后的水量": water,
        "优化后的水溶液E量": solution_e,
        "预测黏度": predicted_viscosity,
        "预测固含量": predicted_solids,
        "总计": total,
        "黏度差": viscosity_difference,
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
            result = run_pso(user_input_values, 预期黏度)
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


import streamlit as st
import pandas as pd
import joblib
from scipy.optimize import minimize


# 加载模型
model_v = joblib.load('viscosity.pkl')
model_s = joblib.load('solids.pkl')

# 优化函数
def optimization_function(params, expected_viscosity):
    """
    params[0]: 水
    params[1]: 水溶液E
    """
    # 替换 '水' 和 '水溶液E' 的值
    modified_features = user_input.copy()
    modified_features['水'] = params[0]
    modified_features['水溶液E'] = params[1]

    # 特征选择
    selected_features_v = modified_features[['乳液A粘度', '乳液F粘度', '水溶液E', '水', '乳液A固含量', '乳液F固含量']]
    selected_features_s = modified_features[['乳液A固含量', '乳液F固含量', '水', '乳液A粘度', '水溶液E', '乳液F粘度']]


    # 预测黏度和理论固含量
    predicted_viscosity = model_v.predict(selected_features_v)[0]
    predicted_solids = model_s.predict(selected_features_s)[0]

    # 逐项计算罚项
    viscosity_penalty = max(0, 4500 - predicted_viscosity) + max(0, predicted_viscosity - 5500)
    viscosity_target_deviation = abs(predicted_viscosity - expected_viscosity)
    solids_penalty = max(0, 0.50 - predicted_solids) + max(0, predicted_solids - 0.54)

    # 总目标函数值
    return viscosity_penalty + solids_penalty + viscosity_target_deviation


# 在 optimize 函数中保持一致
def optimize(user_input_values, expected_viscosity):
    global user_input
    user_input = pd.DataFrame([user_input_values])

    user_input_total = (
        user_input['乳液A'] +
        user_input['乳液F'] +
        user_input['水'] +
        user_input['水溶液E'] +
        user_input['其它']
    )

    # 设置变量范围
    # 初始值和约束
    water_bounds = (10, 100)  # 替换为实际值
    solution_e_bounds = (100, 300)  # 替换为实际值
    initial_guess = [50, 200]


    def total_constraint(params):
        water, solution_e = params
        total = (
            user_input['乳液A'] +
            user_input['乳液F'] +
            water +
            solution_e +
            user_input['其它']
        )
        return total - user_input_total

    constraints = {"type": "ineq", "fun": total_constraint}

    result = minimize(
        optimization_function,
        initial_guess,
        args=(expected_viscosity,),
        bounds=[water_bounds, solution_e_bounds],
        constraints=constraints,
    )

    if result.success:
        optimized_water, optimized_solution_e = result.x

        user_input['水'] = optimized_water
        user_input['水溶液E'] = optimized_solution_e

        selected_features_v = user_input[['乳液A粘度', '乳液F粘度', '水溶液E', '水', '乳液A固含量', '乳液F固含量']]
        selected_features_s = user_input[['乳液A固含量', '乳液F固含量', '水', '乳液A粘度', '水溶液E', '乳液F粘度']]

        # scaled_features_v = scaler_v.transform(selected_features_v)
        # scaled_features_s = scaler_s.transform(selected_features_s)

        predicted_viscosity = model_v.predict(selected_features_v)[0]
        predicted_solids = model_s.predict(selected_features_s)[0]

        total = (
            user_input['乳液A'] +
            user_input['乳液F'] + user_input['水溶液F'] +
            optimized_water +
            optimized_solution_e +
            user_input['其它']
        )

        viscosity_difference = abs(predicted_viscosity - expected_viscosity)
        relative_error = viscosity_difference / expected_viscosity * 100

        return {
            "优化后的水量": optimized_water,
            "优化后的水溶液E量": optimized_solution_e,
            "预测黏度": predicted_viscosity,
            "预测固含量": predicted_solids,
            "总计": total,
            "黏度差": viscosity_difference,
            "相对误差 (%)": relative_error,
        }
    else:
        raise ValueError("优化失败，请检查模型或输入值。")


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
乳液A固含量 = st.sidebar.number_input("乳液A固含量", value=0.555)
乳液F = st.sidebar.number_input("乳液F ", value=1240)
乳液F粘度 = st.sidebar.number_input("乳液F粘度", value=4740)
乳液F固含量 = st.sidebar.number_input("乳液F固含量", value=0.603)
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

    try:
        result = optimize(user_input_values, 预期黏度)
        st.subheader("✨ 优化结果")
        st.success("优化成功！以下是结果：")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("优化后的水量", f"{result['优化后的水量']:.2f} ")
            st.metric("预测黏度", f"{result['预测黏度']:.2f}")
            st.metric("黏度差", f"{result['黏度差']:.2f}")



        with col2:
            st.metric("优化后的水溶液E量", f"{result['优化后的水溶液E量']:.2f} ")
            st.metric("预测固含量", f"{result['预测固含量']:.2f}")

            st.metric("相对误差 (%)", f"{result['相对误差 (%)']:.2f}%")

        st.write(f"### 总计: **<span style='font-size:1.2em'>{result['总计'][0]:.2f} g</span>**", unsafe_allow_html=True)

    except ValueError as e:
        st.error(str(e))


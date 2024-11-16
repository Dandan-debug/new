import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载标准器和模型
scalers = {
    'C': joblib.load('scaler_standard_C.pkl'),
    'P': joblib.load('scaler_standard_P.pkl'),
    'U': joblib.load('scaler_standard_U.pkl')
}

models = {
    'C': joblib.load('rf_C.pkl'),
    'P': joblib.load('xgboost_P.pkl'),
    'U': joblib.load('lightgbm_U.pkl')
}

# 定义特征名称
display_features_to_scale = [
    'Age (years)',                                  # Age (e.g., 52 years)
    'Endometrial thickness (mm)',                   # Endometrial thickness in mm
    'HE4 (pmol/L)',                                 # HE4 level in pmol/L
    'Menopause (1=yes)',                            # Menopause status (1=yes)
    'HRT (Hormone Replacement Therapy, 1=yes)',     # HRT status (1=yes)
    'Endometrial heterogeneity (1=yes)',            # Endometrial heterogeneity (1=yes)
    'Uterine cavity occupation (1=yes)',            # Uterine cavity occupation (1=yes)
    'Uterine cavity occupying lesion with rich blood flow (1=yes)', # Uterine cavity occupying lesion with rich blood flow (1=yes)
    'Uterine cavity fluid (1=yes)'                  # Uterine cavity fluid (1=yes)
]

# 原始特征名称，用于标准化器
original_features_to_scale = [
    'CI_age', 'CI_endometrial thickness', 'CI_HE4', 'CI_menopause',
    'CI_HRT', 'CI_endometrial heterogeneity',
    'CI_uterine cavity occupation',
    'CI_uterine cavity occupying lesion with rich blood flow',
    'CI_uterine cavity fluid'
]

# 额外特征名称映射（添加 .0 后缀）
additional_features = {
    'C': [f"{feature}.0" for feature in ['CM628', 'CM7439', 'CM2345', 'CM1797', 'CM7441', 'CM1348', 'CM2856', 'CM6160',
                                         'CM4009', 'CM7752', 'CM1933', 'CM393', 'CM8446', 'CM4590', 'CM4190', 'CM6181',
                                         'CM4289', 'CM143', 'CM5010', 'CM4900', 'CM6956', 'CM3402', 'CM3601', 'CM1795',
                                         'CM2695', 'CM1895', 'CM6706', 'CM3429', 'CM8692', 'CM6547']],
    'P': [f"{feature}.0" for feature in ['PM469', 'PP29', 'PP202', 'PM224', 'PP121', 'PM883', 'PM323', 'PP61', 'PP464',
                                         'PM285', 'PM846', 'PP435', 'PP41', 'PM102', 'PM477', 'PP629', 'PM867', 'PM168',
                                         'PP610', 'PP344', 'PM722', 'PM446', 'PM339', 'PM794', 'PP590', 'PP220', 'PM302',
                                         'PP526', 'PP24', 'PM631', 'PM38', 'PM267', 'PM893', 'PM472', 'PM501', 'PM241',
                                         'PM857', 'PM733', 'PP28', 'PM526', 'PP649', 'PM101']],
    'U': [f"{feature}.0" for feature in ['UM7578', 'UM510', 'UM507', 'UM670', 'UM351', 'UM5905', 'UM346', 'UM355',
                                         'UM8899', 'UM1152', 'UM5269', 'UM6437', 'UM5906', 'UM7622', 'UM8898', 'UM2132',
                                         'UM3513', 'UM790', 'UM8349', 'UM2093', 'UM4210', 'UM3935', 'UM4256']]
}

# Streamlit界面
st.title("子宫内膜癌风险预测器")

# 模型选择
selected_models = st.multiselect(
    "选择要使用的模型（可以选择一个或多个）",
    options=['U', 'C', 'P'],
    default=['U']
)

# 获取用户输入
user_input = {}

# 定义通用特征输入
for i, feature in enumerate(display_features_to_scale):
    if "1=yes" in feature:  # 对于分类变量，限制输入为0或1
        user_input[original_features_to_scale[i]] = st.selectbox(f"{feature}:", options=[0, 1])
    else:  # 对于连续变量，使用数值输入框
        user_input[original_features_to_scale[i]] = st.number_input(f"{feature}:", min_value=0.0, value=0.0)

# 为每个选定的模型定义额外特征输入
for model_key in selected_models:
    for feature in additional_features[model_key]:
        # 去掉 .0 后缀，仅用于显示
        display_feature = feature.replace(".0", "")
        # 显示为去掉后缀的名称，但仍然保存为带 .0 后缀的键
        user_input[feature] = st.number_input(f"{display_feature} ({model_key}):", min_value=0.0, format="%.9f")

# 预测按钮
if st.button("预测"):
    # 定义模型预测结果存储字典
    model_predictions = {}

    # 对选定的每个模型进行标准化和预测
    for model_key in selected_models:
        # 针对每个模型构建专用的输入数据
        model_input_df = pd.DataFrame([user_input])
        
        # 获取模型所需的特征列
        model_features = original_features_to_scale + additional_features[model_key]
        
        # 仅保留当前模型需要的特征
        model_input_df = model_input_df[model_features]
        
        # 对需要标准化的特征进行标准化
        model_input_df[original_features_to_scale] = scalers[model_key].transform(model_input_df[original_features_to_scale])
        
        # 使用模型进行预测
        predicted_proba = models[model_key].predict_proba(model_input_df)[0]
        predicted_class = models[model_key].predict(model_input_df)[0]
        
        # 保存预测结果
        model_predictions[model_key] = {
            'proba': predicted_proba,
            'class': predicted_class
        }
        
        # 输出每个模型的具体预测概率
        location = {"U": "宫腔", "C": "宫颈", "P": "血浆"}[model_key]
        st.write(f"**{location}筛查模型的预测概率:**")
        st.write(f"- 类别 0（无癌症风险）: {predicted_proba[0]:.2f}")
        st.write(f"- 类别 1（癌症风险）: {predicted_proba[1]:.2f}")

    # 文案输出
    def generate_output(model_key, predicted_class, predicted_proba):
        location = {"U": "宫腔", "C": "宫颈", "P": "血浆"}[model_key]
        risk_level = "癌症风险系数较高" if predicted_class == 1 else "癌症风险系数较低"
        proba_percentage = predicted_proba[1] * 100
        if predicted_class == 1:
            return f"子宫内膜癌{location}筛查模型提示“{risk_level}”，预测癌症概率为{proba_percentage:.1f}%。建议患者进一步接受专科诊断，以便尽早排除风险或启动干预治疗。"
        else:
            return f"子宫内膜癌{location}筛查模型提示“{risk_level}”，预测癌症概率为{proba_percentage:.1f}%。建议患者密切随访或进一步接受专科诊断。同时注意保持健康体重、均衡饮食和规律运动，以预防子宫内膜癌。"

    # 最终结果判定
    if len(selected_models) == 1:
        # 若选择一个模型
        model_key = selected_models[0]
        st.write(generate_output(model_key, model_predictions[model_key]['class'], model_predictions[model_key]['proba']))

    elif len(selected_models) == 2:
        # 若选择两个模型，按排名优先级 U > C > P 确定最终输出
        if 'U' in selected_models:
            model_key = 'U'
        elif 'C' in selected_models:
            model_key = 'C'
        else:
            model_key = 'P'
        st.write(generate_output(model_key, model_predictions[model_key]['class'], model_predictions[model_key]['proba']))

    elif len(selected_models) == 3:
        # 若选择三个模型，根据投票决定最终输出
        cancer_classes = sum(model_predictions[model_key]['class'] for model_key in selected_models)
        if cancer_classes >= 2:
            # 两个或以上模型预测为癌症
            st.write("**子宫内膜癌投票模型提示“癌症风险系数较高”。**")
        else:
            # 两个或以上模型预测为非癌症
            st.write("**子宫内膜癌投票模型提示“癌症风险系数低”。**")
        
        # 输出每个模型的预测概率和建议
        for model_key in selected_models:
            location = {"U": "宫腔", "C": "宫颈", "P": "血浆"}[model_key]
            risk_level = "高" if model_predictions[model_key]['class'] == 1 else "低"
            proba_percentage = model_predictions[model_key]['proba'][1] * 100
            st.write(f"{location}筛查模型预测癌症概率为{proba_percentage:.1f}%。")
        
        # 建议
        if cancer_classes >= 2:
            st.write("建议患者进一步接受确诊检查，以便尽早排除风险或启动干预治疗。")
        else:
            st.write("建议患者密切随访。同时注意保持健康体重、均衡饮食和规律运动，以预防子宫内膜癌。")

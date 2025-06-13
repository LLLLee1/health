import pandas as pd
import numpy as np
import joblib
import streamlit as st
import re
import jieba
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# 1. 数据加载器
class HealthDataLoader:
    def __init__(self, file_path="medical_claims_20250613_162711.csv"):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            st.success(f"✅ 成功加载数据集: {len(self.data)}条健康声明")
            return True
        except Exception as e:
            st.error(f"数据加载失败: {str(e)}")
            return False
        
    def get_sample_data(self, n=5):
        return self.data.sample(n)[['claim', 'credibility']]

# 2. 专业特征工程（结合医学知识）
class HealthFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # 可信度信号（医学专业术语）
        self.reliable_terms = [
            '临床试验', '循证医学', '双盲研究', '随机对照试验', 
            '系统性评价', '学术期刊', '医学报告', '病理研究',
            '药理作用', '临床观察', '流行病学研究', '多中心研究'
        ]
        
        # 风险信号（伪科学特征）
        self.warning_signals = [
            '秘方', '奇迹', '彻底治愈', '永不复发', '包治百病',
            '专家不说', '医院隐藏', '立即见效', '立即转发', '政府隐瞒',
            '效果惊人', '神奇疗效', '祖传秘方', '绝对安全', '纯天然'
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """提取高级健康声明特征"""
        features = []
        
        for text in X:
            # 基础文本特征
            text_length = len(text)
            char_count = len(re.findall(r'[A-Za-z0-9]', text))
            zh_char_count = len(re.findall(r'[\u4e00-\u9fff]', text))
            
            # 可信度特征
            reliable_count = sum(1 for term in self.reliable_terms if term in text)
            warning_count = sum(1 for signal in self.warning_signals if signal in text)
            
            # 情感特征（简化版）
            positive_terms = ['有益', '推荐', '促进', '改善', '提升', '保护']
            negative_terms = ['有害', '避免', '风险', '副作用', '危险', '禁忌']
            positive_score = sum(1 for term in positive_terms if term in text)
            negative_score = sum(1 for term in negative_terms if term in text)
            
            # 数值特征
            has_number = 1 if re.search(r'\d+', text) else 0
            percent_count = text.count('%')
            
            # 结构特征
            sentence_count = text.count('。') + text.count('！') + text.count('？') + 1
            
            # 组合特征
            feature_vec = [
                text_length,
                char_count,
                zh_char_count,
                reliable_count,
                warning_count,
                positive_score,
                negative_score,
                has_number,
                percent_count,
                sentence_count
            ]
            features.append(feature_vec)
            
        return np.array(features)

# 3. 双模型训练管道（文本+特征融合）
class HealthKnowledgePipeline:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.vectorizer = None
        self.feature_engineer = HealthFeatureEngineer()
        self.performance = None
        
    def train_model(self):
        """训练专业健康知识模型"""
        # 准备数据
        X = self.data['claim']
        y = self.data['credibility'].apply(lambda x: 1 if x > 0.7 else 0)  # 1为可信，0为不可信
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 创建模型管道
        self.model = Pipeline([
            ('features', self.feature_engineer),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            ))
        ])
        
        # 训练模型
        with st.spinner("模型训练中，请稍候..."):
            self.model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # 性能指标
            self.performance = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # 保存模型
            joblib.dump(self.model, 'health_knowledge_model.pkl')
            st.success("模型训练完成并保存！")
            st.write(f"测试集准确率: {accuracy:.2f}")
            
        return self.performance
    
    def visualize_performance(self):
        """可视化模型性能"""
        if not self.performance:
            return
            
        cm = self.performance['confusion_matrix']
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['不可信', '可信'], 
                   yticklabels=['不可信', '可信'])
        ax.set_xlabel('预测')
        ax.set_ylabel('真实')
        ax.set_title('混淆矩阵')
        st.pyplot(fig)
        
        st.text("分类报告:")
        st.text(self.performance['classification_report'])

# 4. 专业评估报告生成
class HealthCredibilityReport:
    def __init__(self):
        self.explanation_db = {
            "reliable_terms": "包含可信度信号: 学术研究、临床试验等专业词汇",
            "warning_signals": "检测到风险信号: 秘方、奇迹等伪科学词汇",
            "evidence_based": "包含数据支撑: 有具体数值或百分比",
            "unrealistic": "检测到不切实际的承诺: 彻底治愈、永不复发等"
        }
        
    def generate_report(self, claim, prediction, explanation, credibility_score):
        """生成专业健康声明评估报告"""
        # 可信度等级
        risk_level = "低风险"
        risk_color = "green"
        if credibility_score < 30:
            risk_level = "高风险"
            risk_color = "red"
        elif credibility_score < 70:
            risk_level = "中等风险"
            risk_color = "orange"
        
        # 关键影响因素
        key_factors = []
        for factor, desc in self.explanation_db.items():
            if factor in explanation:
                key_factors.append(desc)
        
        # 医疗专业建议
        recommendations = []
        if prediction == 1:  # 可信
            if credibility_score > 90:
                recommendations.append("✅ 可信度极高，可作为可靠健康参考")
            else:
                recommendations.append("⚠️ 信息基本可信，建议确认最新医学指南")
        else:  # 不可信
            recommendations.append("❌ 可信度不足，请勿直接采纳")
            recommendations.append("🔍 建议查询专业医学资源：WHO、国家卫健委等")
        
        return {
            "claim": claim,
            "credibility_score": credibility_score,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "prediction": "可信度高" if prediction == 1 else "存在风险",
            "key_factors": key_factors,
            "recommendations": recommendations,
            "explanation": explanation
        }
    
    def display_report(self, report):
        """在Streamlit中显示专业报告"""
        st.subheader("健康声明评估报告")
        
        # 整体评分
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("可信度评分", f"{report['credibility_score']}分")
            st.markdown(
                f"<div style='background-color:{report['risk_color']};padding:10px;border-radius:5px;'>"
                f"<strong>风险等级:</strong> {report['risk_level']}</div>",
                unsafe_allow_html=True
            )
        
        # 关键因素
        with col2:
            st.subheader("影响因素分析")
            for factor in report.get('key_factors', []):
                st.markdown(f"- {factor}")
        
        # 建议与解释
        st.subheader("专业建议")
        for rec in report.get('recommendations', []):
            if rec.startswith("✅"):
                st.success(rec)
            elif rec.startswith("⚠️"):
                st.warning(rec)
            else:
                st.error(rec)
        
        # 详细解释
        with st.expander("技术分析详情"):
            for key, value in report.get('explanation', {}).items():
                if isinstance(value, list):
                    st.markdown(f"**{key}**:")
                    for item in value[:3]:
                        st.markdown(f"- {item}")
                elif isinstance(value, dict):
                    st.markdown(f"**{key}**:")
                    for k, v in value.items():
                        st.markdown(f"- {k}: {v}")
                else:
                    st.markdown(f"**{key}**: {value}")

# 5. Streamlit应用主函数
def main_health_app():
    st.set_page_config(
        page_title="科学健康知识可信度分析系统",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🩺 科学健康知识可信度分析系统")
    st.markdown("基于15,000条专业健康声明数据集评估健康信息可信度")
    
    # 初始化状态
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # 加载数据
    data_loader = HealthDataLoader()
    if not data_loader.load_data():
        return
    
    # 模型训练部分
    with st.expander("数据集样本"):
        st.table(data_loader.get_sample_data())
    
    if st.button("训练可信度分析模型"):
        model_pipeline = HealthKnowledgePipeline(data_loader.data)
        performance = model_pipeline.train_model()
        st.text(performance['classification_report'])
        
        # 可视化混淆矩阵
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(performance['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=ax)
        ax.set_title('模型混淆矩阵')
        st.pyplot(fig)
    
    # 声明分析界面
    health_claim = st.text_area(
        "输入需要分析的医学声明:", 
        placeholder="例如：每天饮用绿茶可以降低心脏病风险30%",
        height=120
    )
    
    if st.button("分析可信度"):
        try:
            model = joblib.load('health_knowledge_model.pkl')
            report_gen = HealthCredibilityReport()
            
            # 获取特征值
            feature_engineer = model.named_steps['features']
            feature_values = feature_engineer.transform([health_claim])[0]
            
            # 预测可信度
            prediction = model.predict([health_claim])[0]
            prediction_proba = model.predict_proba([health_claim])[0]
            credibility_score = prediction_proba[1] * 100
            
            # 生成报告
            report = report_gen.generate_report(
                claim=health_claim,
                prediction=prediction,
                explanation={"特征值": feature_values.tolist()},
                credibility_score=round(credibility_score, 1)
            )
            
            # 显示报告
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("可信度评分", f"{credibility_score:.1f}分")
                st.progress(credibility_score/100)
                
            with col2:
                st.markdown(f"**风险评估**: <span style='color:{report['risk_color']};'>{report['risk_level']}</span>", 
                           unsafe_allow_html=True)
                for factor in report['key_factors']:
                    st.markdown(f"- {factor}")
            
            # 建议部分
            st.subheader("专业建议")
            for rec in report['recommendations']:
                if rec.startswith("✅"): st.success(rec)
                elif rec.startswith("⚠️"): st.warning(rec)
                else: st.error(rec)
                
            # 保存历史
            st.session_state.history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "claim": health_claim[:100] + "..." if len(health_claim) > 100 else health_claim,
                "score": round(credibility_score, 1)
            })
                
        except Exception as e:
            st.error(f"分析过程出错: {str(e)}")
            st.error("请确保数据集和模型已准备就绪")
    
    # 历史记录侧边栏
    st.sidebar.title("分析历史")
    if st.session_state.history:
        for item in st.session_state.history[-5:]:
            score = item['score']
            color = "green" if score > 70 else "orange" if score > 30 else "red"
            st.sidebar.markdown(
                f"<div style='border-left:4px solid {color};padding:8px;margin:5px;'>"
                f"{item['claim']}<br><small>{item['time']}</small><br>"
                f"<strong>{score:.1f}分</strong></div>", 
                unsafe_allow_html=True
            )
    else:
        st.sidebar.info("暂无分析历史")
    
    # 专业资源区
    st.sidebar.title("权威医学资源")
    st.sidebar.markdown("[世界卫生组织(WHO)](https://www.who.int)")
    st.sidebar.markdown("[中国国家卫健委](http://www.nhc.gov.cn)")
    st.sidebar.markdown("[美国CDC](https://www.cdc.gov)")

if __name__ == "__main__":
    main_health_app()

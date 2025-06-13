import sys
sys.tracebacklimit = 0
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import re
import jieba
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import altair as alt
from wordcloud import WordCloud
import jieba.analyse

# 1. 数据加载器
class HealthDataLoader:
    def __init__(self, file_path="medical_claims_20250613_162711.csv"):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            
            # 确保数据完整性
            if 'credibility' not in self.data.columns:
                raise ValueError("数据集缺少'credibility'列")
                
            # 添加健康声明类别
            self.data['category'] = self.data['claim'].apply(self.classify_claim)
            
            st.success(f"✅ 成功加载数据集: {len(self.data)}条健康声明")
            return True
        except Exception as e:
            st.error(f"数据加载失败: {str(e)}")
            return False
        
    def classify_claim(self, claim):
        """自动分类健康声明到健康领域"""
        # 基于关键词的简单分类
        categories = {
            '心血管健康': ['心脏', '血压', '胆固醇', '血脂', '中风'],
            '营养饮食': ['饮食', '营养', '维生素', '蛋白质', '脂肪', '碳水', '矿物质'],
            '运动健身': ['运动', '锻炼', '健身', '有氧', '肌肉', '力量'],
            '心理健康': ['抑郁', '焦虑', '压力', '情绪', '睡眠', '心理'],
            '慢性病管理': ['糖尿病', '高血压', '关节炎', '管理', '控制', '慢性'],
            '癌症防治': ['癌症', '肿瘤', '抗癌', '转移', '化疗'],
            '传统医学': ['中医', '草药', '针灸', '经络', '平衡', '寒热'],
            '儿科健康': ['儿童', '发育', '疫苗', '喂养', '早教'],
            '老年健康': ['老年', '老龄', '退休', '关节', '认知'],
        }
        
        for category, keywords in categories.items():
            if any(keyword in claim for keyword in keywords):
                return category
        
        return '其他'
    
    def get_categories(self):
        """获取数据中的健康声明类别"""
        if self.data is not None and 'category' in self.data.columns:
            return self.data['category'].value_counts()
        return None
        
    def get_sample_data(self, n=5):
        return self.data.sample(n)[['claim', 'credibility', 'category']]

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
            '效果惊人', '神奇疗效', '祖极客时间传秘方', '绝对安全', '纯天然'
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
        y = self.data['credibility'].apply(lambda x: 1 if x > 0.7 else 0)
        
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
        self.topic_keywords = {
            "心血管健康": ["心脏", "血压", "血脂", "胆固醇", "中风"],
            "营养饮食": ["饮食", "营养", "维生素", "蛋白质", "脂肪"],
            "运动健身": ["运动", "锻炼", "健身", "有氧", "肌肉"],
            "心理健康": ["压力", "抑郁", "焦虑", "睡眠", "极客时间情绪"],
            "慢性病管理": ["糖尿病", "高血压", "关节炎", "管理", "控制"]
        }
        
    def identify_topic(self, claim):
        """识别健康声明的主题领域"""
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in claim for keyword in keywords):
                return topic
        return "其他健康领域"
        
    def generate_report(self, claim, prediction, explanation, credibility_score):
        """生成专业健康声明评估报告"""
        # 识别主题
        health_topic = self.identify_topic(claim)
        
        # 可信度等级
        risk_level = "低风险"
        risk_color = "green"
        risk_explanation = "该声明可信度高，可作为参考"
        if credibility_score < 30:
            risk_level = "高风险"
            risk_color = "red"
            risk_explanation = "高风险声明，请谨慎对待并核实来源"
        elif credibility_score < 70:
            risk_level = "中等风险"
            risk_color = "orange"
            risk_explanation = "中等风险，需进一步验证信息来源"
        
        # 关键影响因素
        key_factors = []
        for factor, desc in self.explanation_db.items():
            if factor in explanation:
                key_factors.append(desc)
        
        # 医疗专业建议
        recommendations = []
        if prediction == 1:  # 可信
            if credibility_score > 90:
                recommendations.append(f"✅ 可信度极高 ({credibility_score}分)，可作为{health_topic}领域的可靠参考")
            else:
                recommendations.append(f"⚠️ 信息基本可信 ({credibility_score}分)，建议确认{health_topic}领域的最新医学指南")
        else:  # 不可信
            recommendations.append(f"❌ 可信度不足 ({credibility_score}分)，请勿直接采纳")
            recommendations.append(f"🔍 建议查询{health_topic}领域的专业医学资源")
        
        return {
            "claim": claim,
            "credibility_score": credibility_score,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_explanation": risk_explanation,
            "prediction": "可信度高" if prediction == 1 else "存在风险",
            "health_topic": health_topic,
            "key_factors": key_factors,
            "recommendations": recommendations,
            "explanation": explanation
        }
    
    def display_report(self, report):
        """在Streamlit中显示专业报告"""
        st.subheader("健康声明评估报告")
        
        # 整体评分
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            st.metric("可信度评分", f"{report['credibility_score']}分")
            st.markdown(
                f"<div style='background-color:{report['risk_color']};padding:10px;border-radius:5px;'>"
                f"<strong>风险等级:</strong> {report['risk_level']}</div>",
                unsafe_allow_html=True
            )
        
        # 健康主题和风险解释
        with col2:
            st.subheader("主题领域")
            st.markdown(f"**{report['health_topic']}**")
            
            st.subheader("风险说明")
            st.info(f"{report['risk_explanation']}")
        
        # 声明显示
        with col3:
            st.caption("被评估声明")
            st.info(f"**{report['claim']}**")
        
        st.divider()
        
        # 关键因素
        with st.expander("🔍 影响因素分析", expanded=True):
            for factor in report.get('key_factors', []):
                st.markdown(f"- {factor}")
            
            # 特征详情
            if 'explanation' in report and '特征极客时间值' in report['explanation']:
                st.subheader("技术特征分析")
                features = [
                    "文本长度", "英文字符数", "中文字符数", "可信术语数", 
                    "风险信号数", "正面情感词", "负面情感词", "包含数字", 
                    "百分比数量", "句子数量"
                ]
                for i, feat in enumerate(features):
                    value = report['explanation']['特征值'][i]
                    st.markdown(f"- **{feat}**: {value}")
        
        # 建议与解释
        st.subheader("专业建议")
        rec_cols = st.columns(2)
        for i, rec in enumerate(report.get('recommendations', [])):
            with rec_cols[i % 2]:
                if rec.startswith("✅"):
                    st.success(rec)
                elif rec.startswith("⚠️"):
                    st.warning(rec)
                else:
                    st.error(rec)
        
        # 相关资源
        with st.expander("🩺 主题健康资源"):
            self.display_health_resources(report['health_topic'])
    
    def display_health_resources(self, topic):
        """显示主题相关健康资源"""
        resources = {
            "心血管健康": [
                ("中国心血管健康联盟", "https://www.csca.org.cn"),
                ("美国心脏协会", "https://www.heart.org"),
                ("心脏健康手册", "https://www.nhlbi.nih.gov/health-topics/all-publications-and-resources")
            ],
            "营养饮食": [
                ("中国营养学会", "https://www.cnsoc.org"),
                ("营养与饮食学会", "https://www.eatright.org"),
                ("健康饮食指南", "https://www.who.int/publications-detail-redirect/9789240063457")
            ],
            "运动健身": [
                ("中国体育科学学会", "http://www.csss.cn"),
                ("美国运动医学会", "https://www.acsm.org"),
                ("运动处方指南", "https://www.health.gov/paguidelines")
            ],
            "心理健康": [
                ("中国心理卫生协会", "http://www.camh.org.cn"),
                ("世界心理卫生联盟", "https://wfme.org"),
                ("心理健康自助手册", "https://www.who.int/publications-detail-redirect/9789240031029")
            ],
            "慢性病管理": [
                ("中国慢性病管理网", "http://www.chronicdisease.org.cn"),
                ("美国慢性病管理协会", "https://www.pcpcc.org"),
                ("慢性病自我管理指南", "https://www.cdc.gov/chronicdisease/index.htm")
            ]
        }
        
        if topic in resources:
            for name, url in resources[topic]:
                st.markdown(f"🔗 [{name}]({url})")
        else:
            st.info("暂无相关专业资源，请查阅通用医学资源")

# 5. 高级功能扩展
class HealthSystemExtensions:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.claim_matrix = self.vectorizer.fit_transform(data['claim'])
        
    def health_risk_assessment(self):
        """多声明健康风险评估"""
        st.subheader("📈 综合健康风险评估")
        st.info("输入多份健康信息，评估整体风险")
        
        claims = st.text_area(
            "输入多个健康声明（每行一条）",
            height=150,
            placeholder="例如:\n每天喝红酒有益心脏健康\n高脂肪饮食会增加心脏病风险\n维生素C可以预防感冒",
            key="multi_claims"
        )
        
        if st.button("评估整体风险"):
            if not claims.strip():
                st.warning("请输入健康声明内容")
                return
                
            claim_list = [c.strip() for c in claims.split("\n") if c.strip()]
            
            # 评估每条声明的风险
            risks = []
            try:
                model = joblib.load('health_knowledge_model.pkl')
                for claim in claim_list:
                    prediction_proba = model.predict_proba([claim])[0]
                    risk_score = prediction_proba[0] * 100  # 不可信的概率作为风险值
                    risks.append(risk_score)
                
                # 总体风险评估
                avg_risk = np.mean(risks)
                max_risk = max(risks)
                
                st.subheader("整体风险评估结果")
                
                col1, col2 = st.columns(2)
                col1.metric("平均风险值", f"{avg_risk:.1f}分")
                col2.metric("最高风险声明", f"{max_risk:.1极客时间f}分")
                
                # 风险可视化
                risk_data = pd.DataFrame({
                    '声明': [f"声明{i+1}" for i in range(len(risks))],
                    '风险值': risks
                })
                
                # 添加风险类别列
                risk_data['风险类别'] = risk_data['风险值'].apply(
                    lambda x: '高风险' if x > 70 
                    else '中风险' if x > 40 
                    else '低风险'
                )
                
                risk_chart = alt.Chart(risk_data).mark_bar().encode(
                    x=alt.X('声明:N', sort='-y'),
                    y=alt.Y('风险值:Q', title='风险值', scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color('风险类别:N', scale=alt.Scale(
                        domain=['低风险', '中风险', '高风险'],
                        range=['green', 'orange', 'red']
                    )),
                    tooltip=['声明', '风险值', '风险类别']
                ).properties(
                    width=600,
                    height=300
                )
                st.altair_chart(risk_chart, use_container_width=True)
                
                # 风险声明分析
                max_risk_index = np.argmax(risks)
                st.warning(f"**最高风险声明**: {claim_list[max_risk_index]} (风险值: {risks[max_risk_index]:.1f}分)")
                
                # 存储评估结果
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.session_state.setdefault('risk_history', []).append({
                    "timestamp": timestamp,
                    "claims": claim_list,
                    "avg_risk": avg_risk,
                    "max_risk": max_risk
                })
                
            except Exception as e:
                st.error(f"风险评估失败: {str(e)}")
        
        # 显示历史评估记录
        if 'risk_history' in st.session_state and st.session_state['risk_history']:
            st.subheader("历史评估记录")
            history_df = pd.DataFrame(st.session_state['risk_history'])
            st.dataframe(history_df.sort_values('timestamp', ascending=False).head(5))
    
    def health_quiz(self):
        """健康知识小测试"""
        st.subheader("🧪 健康知识小测验")
        st.info("测试您的健康知识水平，识别伪科学信息")
        
        # 从数据集中选择问题
        quiz_questions = self.data.sample(3)[['claim', 'credibility', 'explanation']].reset_index(drop=True)
        
        if st.button("生成新测试"):
            st.session_state.quiz_questions = quiz_questions
            st.session_state.user_answers = [None] * len(quiz_questions)
            st.session_state.quiz_submitted = False
        
        if 'quiz_questions' not in st.session_state:
            st.write("点击上方按钮生成测试题")
            return
            
        questions = st.session_state.quiz_questions
        user_answers = st.session_state.user_answers
        submitted = st.session_state.quiz_submitted
        
        for i in range(len(questions)):
            st.subheader(f"问题 {i+1}")
            # 安全访问行数据
            row = questions.iloc[i]
            claim = row['claim']
            st.markdown(f"**健康声明：** {claim}")
            
            if not submitted:
                options = ['非常可信', '比较可信', '不确定', '不太可信', '非常不可信']
                user_answers[i] = st.radio(
                    f"您认为这个声明的可信度如何？",
                    options,
                    key=f"quiz_q{i}"
                )
            else:
                credibility = row['credibility']
                correct_answer = '非常可信' if credibility > 0.8 else '比较可信' if credibility > 0.6 else '不太可信' if credibility > 0.4 else '非常不可信'
                user_answer = user_answers[i]
                
                st.info(f"您的选择: **{user_answer}**")
                if user_answer == correct_answer:
                    st.success(f"✅ 正确！实际可信度: {credibility:.2f}")
                else:
                    st.error(f"❌ 错误，正确选项是: **{correct_answer}** (实际可信度: {credibility:.2f})")
                
                with st.expander("查看解释"):
                    explanation = row['explanation']
                    st.markdown(f"**科学解释：** {explanation}")
        
        if not submitted:
            if st.button("提交测试", type="primary"):
                st.session_state.quiz_submitted = True
                st.experimental_rerun()
        else:
            # 计算得分
            correct_count = 0
            for i in range(len(questions)):
                credibility = questions.iloc[i]['credibility']
                correct_answer = '非常可信' if credibility > 0.8 else '比较可信' if credibility > 0.6 else '不太可信' if credibility > 0.4 else '非常不可信'
                if user_answers[i] == correct_answer:
                    correct_count += 1
            
            score = correct_count / len(questions) * 100
            
            st.success(f"📝 测试完成！您的得分: **{score:.0f}分**")
            if score >= 80:
                st.balloons()
                st.success("🎉 优秀！您对健康知识有很高的辨别能力")
            elif score >= 60:
                st.info("👍 良好！您对健康信息有一定判断能力")
            else:
                st.warning("💡 继续努力！建议多学习健康知识")

# 6. Streamlit应用主函数
def main_health_app():
    st.set_page_config(
        page_title="科学健康知识可信度分析系统",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.example.com/help',
            'Report a bug': "https://www.example.com/bug",
            'About': "# 科学健康知识分析系统 v2.4"
        }
    )
    
    st.title("🩺 科学健康知识可信度分析系统")
    st.caption("基于专业健康声明的可信度分析与知识发现")
    
    # 初始化状态
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('health_topic', '心血管健康')
    
    # 页面选择器
    page = st.sidebar.selectbox(
        "功能菜单",
        ["健康声明分析", "风险评估", "健康小测试"],
        index=0
    )
    
    # 加载数据
    data_loader = HealthDataLoader()
    if not data_loader.load_data():
        return
    
    # 顶部展示健康主题分布
    category_counts = data_loader.get_categories()
    if category_counts is not None:
        with st.container():
            st.subheader("健康主题分布")
            st.bar_chart(category_counts)
    
    # 显示数据集样本
    with st.expander("数据集样本", expanded=False):
        st.dataframe(data_loader.get_sample_data(3))
    
    # 功能页面路由
    if page == "健康声明分析":
        render_analysis_page(data_loader.data)
    elif page == "风险评估":
        render_risk_assessment_page(data_loader.data)
    elif page == "健康小测试":
        render_quiz_page(data_loader.data)
    
    # 侧边栏区域
    with st.sidebar:
        st.divider()
        st.subheader("历史记录")
        if st.session_state.history:
            for i, item in enumerate(st.session_state.history[-3:]):
                risk_color = "green" if item['score'] > 70 else "orange" if item['score'] > 30 else "red"
                with st.expander(f"记录 {i+1} ({item['score']}分)", expanded=False):
                    st.markdown(f"**声明:** {item['claim']}")
                    st.markdown(f"**时间:** {item['time']}")
                    st.markdown(f"**评估:** <span style='color:{risk_color};'>{item['score']}分</span>", 
                               unsafe_allow_html=True)
        else:
            st.info("暂无分析历史")
        
        st.divider()
        st.subheader("权威医学资源")
        st.markdown("- [世界卫生组织 (WHO)](https://www.who.int)")
        st.markdown("- [中国国家卫健委](http://www.nhc.gov.cn)")
        st.markdown("- [美国疾控中心 (CDC)](https://www.cdc.gov)")
        st.markdown("- [PubMed医学文献](https://pubmed.ncbi.nlm.nih.gov)")
        
        st.divider()
        st.caption("系统版本: 2.4 | 更新日期: 2025-06-15")

def render_analysis_page(data):
    """健康声明分析页面"""
    st.header("🔍 健康声明分析")
    
    # 模型训练部分
    if st.button("训练/更新模型", type="primary"):
        model_pipeline = HealthKnowledgePipeline(data)
        performance = model_pipeline.train_model()
        
        # 可视化模型性能
        with st.expander("模型性能详情"):
            model_pipeline.visualize_performance()
    
    # 声明分析界面
    health_claim = st.text_area(
        "输入需要分析的医学声明:", 
        placeholder="例如：每天饮用绿茶可以降低心脏病风险30%",
        height=120
    )
    
    if st.button("分析可信度", type="secondary"):
        if not health_claim:
            st.warning("请输入健康声明内容")
            return
            
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
            report_gen.display_report(report)
                
            # 保存历史
            st.session_state.history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "claim": health_claim[:100] + "..." if len(health_claim) > 100 else health极客时间_claim,
                "score": round(credibility_score, 1)
            })
                
        except Exception as e:
            st.error(f"分析过程出错: {str(e)}")
            st.error("请确保数据集和模型已准备就绪")

def render_risk_assessment_page(data):
    """多声明风险评估页面"""
    st.header("📈 综合健康风险评估")
    extensions = HealthSystemExtensions(data)
    extensions.health_risk_assessment()

def render_quiz_page(data):
    """健康知识小测试页面"""
    st.header("🧪 健康知识小测验")
    extensions = HealthSystemExtensions(data)
    extensions.health_quiz()

# 运行主应用
if __name__ == "__main__":
    main_health_app()

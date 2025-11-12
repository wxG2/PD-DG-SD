"""
PD脑电Diffusion模型实验评估代码框架
包含所有关键实验的完整实现
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats, signal
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import pandas as pd

# =============================================================================
# 实验1：最少样本量实验
# =============================================================================

class MinimalSampleExperiment:
    """确定生成有效PD样本所需的最少输入样本数量"""
    
    def __init__(self, model, train_pd_data, test_pd_data, test_healthy_data):
        self.model = model
        self.train_pd = train_pd_data
        self.test_pd = test_pd_data
        self.test_healthy = test_healthy_data
        
    def run_experiment(self, 
                       input_nums=[1, 2, 3, 5, 7, 10, 15, 20],
                       n_repeats=10,
                       n_generate=100):
        """
        运行最少样本量实验
        
        Returns:
            results: Dict包含每个配置的性能指标
        """
        results = {n: [] for n in input_nums}
        
        for n_input in input_nums:
            print(f"\n测试输入样本数量: {n_input}")
            
            for repeat in range(n_repeats):
                # 随机选择n_input个PD样本作为条件
                indices = np.random.choice(len(self.train_pd), n_input, replace=False)
                condition_samples = [self.train_pd[i] for i in indices]
                
                # 生成样本
                with torch.no_grad():
                    generated = self.model.generate(
                        condition_samples, 
                        n_samples=n_generate
                    )
                
                # 评估
                metrics = self._evaluate_generation(generated)
                results[n_input].append(metrics)
                
                print(f"  重复 {repeat+1}/{n_repeats}: "
                      f"FID={metrics['fid']:.2f}, "
                      f"Acc={metrics['downstream_acc']:.2%}")
        
        # 统计汇总
        summary = self._summarize_results(results)
        self._plot_results(summary)
        self._determine_minimum(summary)
        
        return summary
    
    def _evaluate_generation(self, generated):
        """全面评估生成质量"""
        metrics = {}
        
        # 1. FID分数
        metrics['fid'] = self._compute_fid(generated, self.test_pd)
        
        # 2. Inception Score
        metrics['is_score'] = self._compute_inception_score(generated)
        
        # 3. 频谱相似度
        metrics['spectral_similarity'] = self._compute_spectral_similarity(
            generated, self.test_pd
        )
        
        # 4. 下游分类任务性能
        metrics['downstream_acc'] = self._downstream_classification(generated)
        
        # 5. Beta功率增强
        metrics['beta_enhancement'] = self._compute_beta_enhancement(generated)
        
        # 6. PAC强度
        metrics['pac_strength'] = self._compute_pac(generated)
        
        # 7. 多样性
        metrics['diversity'] = self._compute_diversity(generated)
        
        return metrics
    
    def _compute_fid(self, generated, real):
        """计算Fréchet Inception Distance"""
        # 提取特征
        gen_features = self._extract_features(generated)
        real_features = self._extract_features(real)
        
        # 计算均值和协方差
        mu_gen, sigma_gen = gen_features.mean(0), np.cov(gen_features, rowvar=False)
        mu_real, sigma_real = real_features.mean(0), np.cov(real_features, rowvar=False)
        
        # FID计算
        diff = mu_gen - mu_real
        covmean = scipy.linalg.sqrtm(sigma_gen @ sigma_real)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_gen + sigma_real - 2*covmean)
        return fid
    
    def _compute_inception_score(self, generated):
        """计算Inception Score (基于预训练的EEG分类器)"""
        # 使用预训练的分类器预测
        probs = self.pretrained_classifier.predict_proba(generated)
        
        # 计算IS
        py = probs.mean(axis=0)
        scores = []
        for i in range(len(generated)):
            pyx = probs[i]
            scores.append(stats.entropy(pyx, py))
        
        return np.exp(np.mean(scores))
    
    def _compute_spectral_similarity(self, generated, real):
        """计算频谱相似度"""
        # 计算功率谱密度
        psd_gen = self._compute_psd_batch(generated)
        psd_real = self._compute_psd_batch(real)
        
        # 使用余弦相似度
        similarities = []
        for psd_g in psd_gen:
            sims = [self._cosine_similarity(psd_g, psd_r) for psd_r in psd_real]
            similarities.append(np.max(sims))
        
        return np.mean(similarities)
    
    def _downstream_classification(self, generated):
        """训练分类器并测试性能"""
        # 混合训练集
        train_data = np.vstack([generated, self.train_pd[:len(generated)]])
        train_labels = np.array([1]*len(generated) + [1]*len(generated))
        
        # 训练简单的EEGNet
        classifier = self._train_eegnet(train_data, train_labels)
        
        # 在真实测试集上评估
        test_data = np.vstack([self.test_pd, self.test_healthy])
        test_labels = np.array([1]*len(self.test_pd) + [0]*len(self.test_healthy))
        
        predictions = classifier.predict(test_data)
        accuracy = accuracy_score(test_labels, predictions)
        
        return accuracy
    
    def _compute_beta_enhancement(self, generated):
        """计算beta频段功率增强倍数"""
        beta_band = (12, 30)
        
        # 生成数据的beta功率
        beta_gen = self._compute_band_power_batch(generated, beta_band)
        
        # 真实PD数据的beta功率
        beta_real = self._compute_band_power_batch(self.test_pd, beta_band)
        
        # 增强倍数
        enhancement = beta_gen.mean() / beta_real.mean()
        return enhancement
    
    def _compute_pac(self, generated):
        """计算相位-振幅耦合强度"""
        phase_band = (4, 8)  # theta
        amp_band = (12, 30)  # beta
        
        pac_values = []
        for eeg in generated:
            # 提取相位和振幅
            phase = self._extract_phase(eeg, phase_band)
            amplitude = self._extract_amplitude(eeg, amp_band)
            
            # 计算调制指数
            mi = self._modulation_index(phase, amplitude)
            pac_values.append(mi)
        
        return np.mean(pac_values)
    
    def _compute_diversity(self, generated):
        """计算生成样本的多样性"""
        n_samples = len(generated)
        distances = []
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # 使用DTW距离
                dist = self._dtw_distance(generated[i], generated[j])
                distances.append(dist)
        
        return np.mean(distances)
    
    def _determine_minimum(self, summary):
        """确定最少所需样本数量"""
        print("\n" + "="*60)
        print("最少样本量判定")
        print("="*60)
        
        thresholds = {
            'fid': 50,
            'downstream_acc': 0.75,
            'beta_enhancement': 1.3,
            'pac_strength': 0.10,
            'spectral_similarity': 0.70
        }
        
        for n_input in sorted(summary.keys()):
            metrics = summary[n_input]
            
            # 检查是否满足所有阈值
            checks = {
                'fid': metrics['fid_mean'] < thresholds['fid'],
                'acc': metrics['acc_mean'] > thresholds['downstream_acc'],
                'beta': metrics['beta_mean'] > thresholds['beta_enhancement'],
                'pac': metrics['pac_mean'] > thresholds['pac_strength'],
                'spec': metrics['spec_mean'] > thresholds['spectral_similarity']
            }
            
            passed = sum(checks.values())
            
            print(f"\n输入样本数 = {n_input}:")
            print(f"  满足条件: {passed}/5")
            for key, value in checks.items():
                print(f"  - {key}: {'✓' if value else '✗'}")
            
            if passed >= 4:  # 至少满足4个条件
                print(f"\n>>> 建议最少样本数量: {n_input}")
                return n_input
        
        print("\n警告: 没有配置满足最少4个条件")
        return None

# =============================================================================
# 实验2：学习曲线与质量上限实验
# =============================================================================

class LearningCurveExperiment:
    """确定模型生成质量的上限及所需训练样本量"""
    
    def __init__(self, model_class, train_data, test_data):
        self.model_class = model_class
        self.train_data = train_data
        self.test_data = test_data
    
    def run_experiment(self, 
                       train_sizes=[10, 20, 30, 40, 50, 60, 80, 100, 150, 200],
                       n_folds=5):
        """运行学习曲线实验"""
        results = {size: [] for size in train_sizes}
        
        for train_size in train_sizes:
            print(f"\n训练样本量: {train_size}")
            
            for fold in range(n_folds):
                # 随机采样训练集
                indices = np.random.choice(
                    len(self.train_data), 
                    train_size, 
                    replace=False
                )
                train_subset = [self.train_data[i] for i in indices]
                
                # 训练模型
                model = self._train_model(train_subset)
                
                # 生成并评估
                generated = model.generate(n_samples=500)
                metrics = self._evaluate(generated)
                
                results[train_size].append(metrics)
                print(f"  Fold {fold+1}: FID={metrics['fid']:.2f}")
        
        # 分析结果
        summary = self._analyze_learning_curve(results)
        self._fit_and_predict_asymptote(summary)
        self._detect_saturation_point(summary)
        
        return summary
    
    def _train_model(self, train_data, epochs=200):
        """训练Diffusion模型"""
        model = self.model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            for batch in self._make_batches(train_data, batch_size=32):
                loss = model.train_step(batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return model
    
    def _analyze_learning_curve(self, results):
        """分析学习曲线"""
        summary = {}
        
        for train_size, metrics_list in results.items():
            # 计算均值和标准差
            metrics_array = {
                key: [m[key] for m in metrics_list]
                for key in metrics_list[0].keys()
            }
            
            summary[train_size] = {
                f"{key}_mean": np.mean(values)
                for key, values in metrics_array.items()
            }
            summary[train_size].update({
                f"{key}_std": np.std(values)
                for key, values in metrics_array.items()
            })
        
        return summary
    
    def _fit_and_predict_asymptote(self, summary):
        """拟合学习曲线并预测渐近上限"""
        from scipy.optimize import curve_fit
        
        train_sizes = np.array(sorted(summary.keys()))
        fid_means = np.array([summary[s]['fid_mean'] for s in train_sizes])
        
        # 定义指数饱和模型: y = a - b * exp(-c * x)
        def exp_saturation(x, a, b, c):
            return a - b * np.exp(-c * x)
        
        # 拟合
        try:
            params, covariance = curve_fit(
                exp_saturation, 
                train_sizes, 
                fid_means,
                p0=[15, 50, 0.01],  # 初始猜测
                maxfev=10000
            )
            
            asymptote = params[0]
            
            print("\n" + "="*60)
            print("学习曲线拟合结果")
            print("="*60)
            print(f"预测的FID渐近上限: {asymptote:.2f}")
            print(f"拟合参数: a={params[0]:.2f}, b={params[1]:.2f}, c={params[2]:.4f}")
            
            # 预测达到95%上限所需样本量
            target_fid = asymptote * 1.05  # 95%性能
            required_samples = -np.log((asymptote - target_fid) / params[1]) / params[2]
            print(f"达到95%上限所需样本量: {int(required_samples)}")
            
            # 可视化
            self._plot_learning_curve_with_fit(
                train_sizes, fid_means, exp_saturation, params
            )
            
        except Exception as e:
            print(f"拟合失败: {e}")
    
    def _detect_saturation_point(self, summary):
        """检测性能饱和点"""
        train_sizes = sorted(summary.keys())
        fid_means = [summary[s]['fid_mean'] for s in train_sizes]
        
        # 计算相对改进率
        improvements = []
        for i in range(1, len(fid_means)):
            improvement = (fid_means[i-1] - fid_means[i]) / fid_means[i-1]
            improvements.append(improvement)
        
        # 检测饱和点（连续3个点改进率<1%）
        saturation_point = None
        for i in range(len(improvements) - 2):
            if all(imp < 0.01 for imp in improvements[i:i+3]):
                saturation_point = train_sizes[i+1]
                break
        
        print("\n" + "="*60)
        print("饱和点分析")
        print("="*60)
        if saturation_point:
            print(f"检测到FID饱和点: {saturation_point} 个样本")
            print(f"饱和点FID值: {summary[saturation_point]['fid_mean']:.2f}")
        else:
            print("未检测到明显饱和点，建议增加训练样本")

# =============================================================================
# 实验3：临床相关性验证
# =============================================================================

class ClinicalValidationExperiment:
    """临床专家盲评实验"""
    
    def __init__(self, generated_pd, real_pd, real_healthy):
        self.generated_pd = generated_pd
        self.real_pd = real_pd
        self.real_healthy = real_healthy
    
    def prepare_blinded_evaluation(self, n_samples=50):
        """准备盲评样本"""
        # 随机选择样本
        gen_indices = np.random.choice(len(self.generated_pd), n_samples, replace=False)
        real_pd_indices = np.random.choice(len(self.real_pd), n_samples, replace=False)
        healthy_indices = np.random.choice(len(self.real_healthy), n_samples, replace=False)
        
        # 构建评估集
        eval_samples = []
        eval_labels = []  # 仅用于后续分析
        
        for i in gen_indices:
            eval_samples.append(self.generated_pd[i])
            eval_labels.append('generated_pd')
        
        for i in real_pd_indices:
            eval_samples.append(self.real_pd[i])
            eval_labels.append('real_pd')
        
        for i in healthy_indices:
            eval_samples.append(self.real_healthy[i])
            eval_labels.append('healthy')
        
        # 随机打乱
        shuffle_indices = np.random.permutation(len(eval_samples))
        eval_samples = [eval_samples[i] for i in shuffle_indices]
        eval_labels = [eval_labels[i] for i in shuffle_indices]
        
        # 保存用于专家评估
        self._save_for_experts(eval_samples, eval_labels)
        
        return eval_samples, eval_labels
    
    def analyze_expert_ratings(self, expert_ratings):
        """分析专家评分"""
        # expert_ratings: List[Dict] where each dict contains ratings from one expert
        
        # 1. 计算专家间一致性 (Cohen's Kappa)
        kappa_scores = self._compute_inter_rater_agreement(expert_ratings)
        
        # 2. 计算生成样本被识别为"真实"的比例
        authenticity_rate = self._compute_authenticity_rate(expert_ratings)
        
        # 3. 计算PD特征识别的灵敏度和特异度
        sensitivity, specificity = self._compute_diagnostic_accuracy(expert_ratings)
        
        print("\n" + "="*60)
        print("临床专家评估结果")
        print("="*60)
        print(f"专家间一致性 (Fleiss' Kappa): {kappa_scores['fleiss']:.3f}")
        print(f"生成样本真实性得分: {authenticity_rate:.2%}")
        print(f"PD特征识别灵敏度: {sensitivity:.2%}")
        print(f"PD特征识别特异度: {specificity:.2%}")
        
        return {
            'kappa': kappa_scores,
            'authenticity': authenticity_rate,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

# =============================================================================
# 实验4：多样性与模式崩塌检测
# =============================================================================

class DiversityAnalysisExperiment:
    """评估生成样本的多样性"""
    
    def __init__(self, model):
        self.model = model
    
    def evaluate_diversity(self, generated_samples):
        """全面评估多样性"""
        results = {}
        
        # 1. 成对距离分析
        results['pairwise_distances'] = self._compute_pairwise_distances(
            generated_samples
        )
        
        # 2. 特征空间覆盖率
        results['coverage'] = self._compute_coverage(generated_samples)
        
        # 3. 频谱多样性
        results['spectral_diversity'] = self._compute_spectral_diversity(
            generated_samples
        )
        
        # 4. 聚类一致性
        results['cluster_consistency'] = self._analyze_clustering(
            generated_samples
        )
        
        # 5. 时间动态多样性
        results['temporal_diversity'] = self._compute_temporal_diversity(
            generated_samples
        )
        
        return results
    
    def detect_mode_collapse(self, n_trials=20, n_samples_per_trial=50):
        """检测模式崩塌"""
        diversities = []
        
        for trial in range(n_trials):
            # 使用不同随机种子
            np.random.seed(trial)
            torch.manual_seed(trial)
            
            # 生成样本
            generated = self.model.generate(n_samples=n_samples_per_trial)
            
            # 计算该批次的多样性
            diversity = self._compute_intra_batch_diversity(generated)
            diversities.append(diversity)
        
        # 分析
        diversity_mean = np.mean(diversities)
        diversity_std = np.std(diversities)
        cv = diversity_std / diversity_mean  # 变异系数
        
        print("\n" + "="*60)
        print("模式崩塌检测")
        print("="*60)
        print(f"平均多样性: {diversity_mean:.4f}")
        print(f"多样性标准差: {diversity_std:.4f}")
        print(f"变异系数: {cv:.4f}")
        
        if cv < 0.1:
            print("⚠️  警告: 检测到潜在的模式崩塌（CV < 0.1）")
        else:
            print("✓  未检测到模式崩塌")
        
        return {
            'mean': diversity_mean,
            'std': diversity_std,
            'cv': cv,
            'collapsed': cv < 0.1
        }

# =============================================================================
# 实验5：可解释性分析
# =============================================================================

class InterpretabilityExperiment:
    """分析生成过程中的特征增强"""
    
    def analyze_feature_enhancement(self, real_pd, generated_pd, real_healthy):
        """分析哪些特征被增强"""
        
        features = self._extract_all_features(real_pd, generated_pd, real_healthy)
        
        print("\n" + "="*60)
        print("特征增强分析")
        print("="*60)
        
        for feature_name in features.keys():
            real_values = features[feature_name]['real_pd']
            gen_values = features[feature_name]['generated_pd']
            healthy_values = features[feature_name]['healthy']
            
            # 增强倍数
            enhancement = gen_values.mean() / real_values.mean()
            
            # 统计显著性
            t_stat, p_value = stats.ttest_ind(gen_values, real_values)
            
            # Cohen's d (效应量)
            cohens_d = (gen_values.mean() - real_values.mean()) / \
                      np.sqrt((gen_values.std()**2 + real_values.std()**2) / 2)
            
            # 与健康对照的分离度
            t_stat_hc, p_value_hc = stats.ttest_ind(gen_values, healthy_values)
            
            print(f"\n{feature_name}:")
            print(f"  增强倍数: {enhancement:.2f}x")
            print(f"  vs真实PD p值: {p_value:.4f} {'(ns)' if p_value > 0.05 else '(*)'}")
            print(f"  效应量: {cohens_d:.2f}")
            print(f"  vs健康对照 p值: {p_value_hc:.4e}")

# =============================================================================
# 主实验运行器
# =============================================================================

class ExperimentRunner:
    """统一管理所有实验"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def run_all_experiments(self):
        """按顺序运行所有实验"""
        
        print("开始实验流程...")
        print("="*80)
        
        # 实验1: 最少样本量
        print("\n[1/5] 运行最少样本量实验...")
        exp1 = MinimalSampleExperiment(
            self.config['model'],
            self.config['train_pd'],
            self.config['test_pd'],
            self.config['test_healthy']
        )
        self.results['minimal_samples'] = exp1.run_experiment()
        
        # 实验2: 学习曲线
        print("\n[2/5] 运行学习曲线实验...")
        exp2 = LearningCurveExperiment(
            self.config['model_class'],
            self.config['train_data'],
            self.config['test_data']
        )
        self.results['learning_curve'] = exp2.run_experiment()
        
        # 实验3: 临床验证
        print("\n[3/5] 准备临床验证...")
        exp3 = ClinicalValidationExperiment(
            self.results['generated_pd'],
            self.config['real_pd'],
            self.config['real_healthy']
        )
        self.results['clinical'] = exp3.prepare_blinded_evaluation()
        
        # 实验4: 多样性分析
        print("\n[4/5] 运行多样性分析...")
        exp4 = DiversityAnalysisExperiment(self.config['model'])
        self.results['diversity'] = exp4.evaluate_diversity(
            self.results['generated_pd']
        )
        self.results['mode_collapse'] = exp4.detect_mode_collapse()
        
        # 实验5: 可解释性
        print("\n[5/5] 运行可解释性分析...")
        exp5 = InterpretabilityExperiment()
        exp5.analyze_feature_enhancement(
            self.config['real_pd'],
            self.results['generated_pd'],
            self.config['real_healthy']
        )
        
        # 生成最终报告
        self.generate_final_report()
        
        print("\n" + "="*80)
        print("所有实验完成!")
        
        return self.results
    
    def generate_final_report(self):
        """生成最终实验报告"""
        report = f"""
        ================================================================
        PD脑电Diffusion模型实验总结报告
        ================================================================
        
        1. 最少样本量: {self.results['minimal_samples']['minimum']} 个
           - FID: {self.results['minimal_samples']['fid']:.2f}
           - 下游准确率: {self.results['minimal_samples']['accuracy']:.2%}
        
        2. 质量上限:
           - 预测FID上限: {self.results['learning_curve']['asymptote']:.2f}
           - 达到95%性能所需样本: {self.results['learning_curve']['required_samples']}
           - 饱和点: {self.results['learning_curve']['saturation_point']} 个样本
        
        3. 临床验证:
           - 专家间一致性: {self.results['clinical']['kappa']:.3f}
           - 真实性评分: {self.results['clinical']['authenticity']:.2%}
        
        4. 多样性:
           - 平均成对距离: {self.results['diversity']['pairwise_distances']:.4f}
           - 覆盖率: {self.results['diversity']['coverage']:.2%}
           - 模式崩塌: {'未检测到' if not self.results['mode_collapse']['collapsed'] else '⚠️ 检测到'}
        
        ================================================================
        """
        
        print(report)
        
        # 保存到文件
        with open('experiment_report.txt', 'w') as f:
            f.write(report)

if __name__ == "__main__":
    # 使用示例
    config = {
        'model': None,  # 你的Diffusion模型实例
        'model_class': None,  # 模型类
        'train_pd': None,  # 训练集PD数据
        'test_pd': None,  # 测试集PD数据
        'test_healthy': None,  # 测试集健康数据
        'train_data': None,  # 完整训练数据
        'test_data': None,  # 完整测试数据
        'real_pd': None,  # 真实PD数据
        'real_healthy': None  # 真实健康数据
    }
    
    runner = ExperimentRunner(config)
    results = runner.run_all_experiments()

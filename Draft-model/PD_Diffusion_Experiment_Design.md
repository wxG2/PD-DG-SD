# PD脑电Diffusion模型实验设计方案

## 一、确定最少有效生成样本量的实验

### 实验1.1：输入样本数量敏感性分析（N-Shot Generation）

**目标**：确定生成有效PD样本所需的最少输入样本数量

**实验设计**：
```python
# 实验参数设置
input_sample_nums = [1, 2, 3, 5, 7, 10, 15, 20]  # 输入的真实PD样本数量
n_repeats = 10  # 每个配置重复10次实验
n_generate_per_config = 100  # 每次生成100个样本
```

**实验步骤**：
1. **数据准备**
   - 从训练集中随机抽取N个PD样本作为条件输入
   - 保持测试集固定，确保评估的一致性

2. **生成与评估**
   ```python
   for n_input in input_sample_nums:
       for repeat in range(n_repeats):
           # 随机选择n_input个PD样本
           condition_samples = random.sample(train_pd_set, n_input)
           
           # 生成样本
           generated = model.generate(condition_samples, n_samples=100)
           
           # 多维度评估
           metrics = {
               'fid_score': compute_fid(generated, real_pd_test),
               'is_score': compute_inception_score(generated),
               'spectral_similarity': compute_spectral_distance(generated, real_pd),
               'clinical_validity': clinical_expert_score(generated),
               'downstream_acc': train_classifier_and_test(generated, real_pd_test)
           }
   ```

3. **判定标准**
   - **有效性阈值**：
     - FID < 50（相比真实PD样本）
     - 下游分类准确率 > 75%
     - 临床专家盲评准确率 > 70%
     - Beta功率比真实PD均值的±20%范围内
   
   - **最少样本量定义**：
     满足上述所有标准的最小N值，且在10次重复中至少8次达标

**预期结果**：
- 绘制性能-输入样本量曲线
- 确定"拐点"——性能提升开始变缓的点
- 论文中报告：最少N样本，95%置信区间

---

## 二、确定生成质量上限的实验

### 实验2.1：学习曲线与饱和度分析

**目标**：确定模型生成质量的理论上限及达到上限所需的训练样本量

**实验设计**：
```python
# 训练样本量梯度
train_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
               120, 150, 200, 250, 300, 'all']  # 或实际总数的百分比

# 对每个训练规模
for train_size in train_sizes:
    # 5折交叉验证确保结果稳定性
    for fold in range(5):
        # 训练模型
        model = train_diffusion_model(train_data[:train_size], epochs=200)
        
        # 生成测试样本
        generated = model.generate(n_samples=500)
        
        # 全面评估
        evaluate_generation_quality(generated, real_test_data)
```

**评估指标体系**：

#### 2.1.1 生成质量指标
```python
quality_metrics = {
    # 分布相似性
    'fid': compute_fid(generated, real),
    'mmd': compute_mmd(generated, real),  # Maximum Mean Discrepancy
    'kl_divergence': compute_kl_div(generated, real),
    
    # 多样性指标
    'intra_diversity': compute_pairwise_distance(generated),
    'coverage': compute_feature_space_coverage(generated, real),
    
    # 真实性指标
    'authenticity_score': discriminator_score(generated)
}
```

#### 2.1.2 生理有效性指标
```python
physiological_metrics = {
    # 频谱特征
    'beta_power_ratio': compute_beta_enhancement(generated),
    'theta_alpha_ratio': compute_theta_alpha_ratio(generated),
    'delta_power': compute_delta_power(generated),
    
    # 时频特征
    'pac_strength': compute_pac(generated, phase_band=(4,8), amp_band=(12,30)),
    'beta_burst_dynamics': compute_burst_metrics(generated),
    
    # 空间特征
    'motor_cortex_beta': compute_regional_power(generated, 'motor'),
    'spatial_coherence': compute_inter_channel_coherence(generated),
    
    # 连接性特征
    'plv': compute_phase_locking_value(generated),  # Phase Locking Value
    'wpli': compute_weighted_pli(generated),  # Weighted Phase Lag Index
}
```

#### 2.1.3 统计显著性检验
```python
def statistical_validation(generated, real_pd, real_healthy):
    """验证生成数据的统计特性"""
    
    # 1. 与真实PD的相似性（应该不显著）
    t_stat_pd, p_value_pd = stats.ttest_ind(
        extract_features(generated),
        extract_features(real_pd)
    )
    
    # 2. 与健康对照的差异性（应该显著）
    t_stat_hc, p_value_hc = stats.ttest_ind(
        extract_features(generated),
        extract_features(real_healthy)
    )
    
    # 3. KS检验（分布一致性）
    ks_stat, ks_p = stats.ks_2samp(
        generated.flatten(),
        real_pd.flatten()
    )
    
    return {
        'vs_real_pd_pvalue': p_value_pd,  # 期望 > 0.05
        'vs_healthy_pvalue': p_value_hc,  # 期望 < 0.05
        'ks_pvalue': ks_p  # 期望 > 0.05
    }
```

**上限判定方法**：

1. **拟合学习曲线**
   ```python
   from scipy.optimize import curve_fit
   
   # 使用幂律或对数函数拟合
   def learning_curve(x, a, b, c):
       return a - b * np.exp(-c * x)  # 指数饱和模型
   
   # 拟合参数
   params, _ = curve_fit(learning_curve, train_sizes, fid_scores)
   
   # 预测的渐近上限
   asymptote = params[0]
   ```

2. **饱和点检测**
   ```python
   def detect_saturation_point(train_sizes, metrics):
       """检测性能饱和点"""
       improvements = np.diff(metrics) / metrics[:-1]  # 相对改进率
       
       # 当连续3个点改进率 < 1%时认为饱和
       for i in range(len(improvements) - 2):
           if all(improvements[i:i+3] < 0.01):
               return train_sizes[i]
       
       return train_sizes[-1]  # 未饱和
   ```

**预期结果**：
- 学习曲线图（训练样本量 vs 各项指标）
- 饱和点分析：FID饱和点在120-150样本，下游任务准确率饱和点在200-250样本
- 理论上限：FID ≈ 15-20（基于拟合曲线外推）

---

## 三、下游任务性能评估实验

### 实验3.1：数据增强效果验证

**目标**：验证生成数据是否能提升分类器性能

**实验设计**：
```python
augmentation_ratios = [0, 0.5, 1.0, 2.0, 5.0, 10.0]  # 生成数据/真实数据比例
classifiers = ['EEGNet', 'DeepConvNet', 'ShallowConvNet', 'CNN-LSTM']

for ratio in augmentation_ratios:
    n_real = len(real_train_pd)
    n_generated = int(n_real * ratio)
    
    # 生成增强数据
    generated_pd = model.generate(n_samples=n_generated)
    
    # 混合训练集
    train_set = combine(real_train_pd, generated_pd)
    
    for clf_name in classifiers:
        # 训练分类器
        classifier = train_classifier(clf_name, train_set)
        
        # 在真实测试集上评估
        results = evaluate(classifier, real_test_set)
        
        # 跨数据集泛化测试
        cross_dataset_results = evaluate(classifier, other_dataset)
```

**关键评估**：
- **同数据集性能**：Sande训练→Sande测试
- **跨数据集泛化**：Sande训练→UNM测试
- **LOSO验证**：Leave-One-Subject-Out
- **类别平衡性**：检查是否只是增加了样本量的效果

---

## 四、生成样本临床相关性实验

### 实验4.1：专家盲评实验

**设计**：
```python
# 准备评估集
evaluation_set = {
    'real_pd': 50,        # 真实PD样本
    'generated_pd': 50,   # 生成PD样本
    'real_healthy': 50    # 真实健康对照
}

# 随机化并盲化
blinded_samples = shuffle_and_blind(evaluation_set)

# 邀请3-5位神经科医生/脑电专家
experts = ['expert_1', 'expert_2', 'expert_3']

for expert in experts:
    ratings = expert.evaluate(blinded_samples, criteria=[
        'is_realistic',           # 是否像真实EEG (1-5分)
        'shows_pd_features',      # 是否有PD特征 (1-5分)
        'feature_prominence',     # 特征显著性 (1-5分)
        'classification'          # 分类为PD/健康/无效
    ])
```

**分析方法**：
- Cohen's Kappa（专家间一致性）
- 灵敏度/特异度（是否能正确识别PD）
- 生成样本被识别为"假"的比例（应该低）

### 实验4.2：UPDRS评分相关性分析

**设计**：
```python
# 如果原始数据有UPDRS评分
for subject in pd_subjects:
    updrs_score = subject.clinical_scores['UPDRS_III']
    real_eeg = subject.eeg_data
    
    # 生成相似样本
    generated = model.generate(condition=[real_eeg])
    
    # 提取特征
    real_features = extract_pd_features(real_eeg)
    gen_features = extract_pd_features(generated)
    
    # 相关性分析
    correlation_real = pearsonr(real_features['beta_power'], updrs_score)
    correlation_gen = pearsonr(gen_features['beta_power'], updrs_score)
    
    # 生成样本应保持与UPDRS的相关性
    assert abs(correlation_real - correlation_gen) < threshold
```

---

## 五、模式崩塌与多样性评估

### 实验5.1：生成多样性分析

```python
def evaluate_diversity(generated_samples):
    """评估生成样本的多样性"""
    
    # 1. 成对距离分析
    pairwise_distances = []
    for i in range(len(generated_samples)):
        for j in range(i+1, len(generated_samples)):
            dist = compute_eeg_distance(
                generated_samples[i], 
                generated_samples[j],
                metric='dtw'  # Dynamic Time Warping
            )
            pairwise_distances.append(dist)
    
    diversity_score = np.mean(pairwise_distances)
    
    # 2. 特征空间覆盖率
    real_features = extract_features(real_pd_samples)
    gen_features = extract_features(generated_samples)
    
    # 计算生成样本在真实特征空间中的覆盖率
    coverage = compute_coverage(gen_features, real_features)
    
    # 3. 频谱多样性
    spectral_diversity = compute_spectral_diversity(generated_samples)
    
    # 4. 聚类分析
    n_clusters_real = optimal_clusters(real_pd_samples)
    n_clusters_gen = optimal_clusters(generated_samples)
    
    return {
        'diversity_score': diversity_score,
        'coverage': coverage,
        'spectral_diversity': spectral_diversity,
        'cluster_consistency': n_clusters_gen / n_clusters_real
    }
```

### 实验5.2：模式崩塌检测

```python
def detect_mode_collapse(model, n_trials=20):
    """检测是否存在模式崩塌"""
    
    results = []
    for trial in range(n_trials):
        # 使用不同的随机种子和条件样本
        set_seed(trial)
        condition_samples = random.sample(train_pd, k=5)
        
        # 生成样本
        generated = model.generate(condition_samples, n_samples=50)
        
        # 计算该批次的多样性
        diversity = compute_intra_batch_diversity(generated)
        results.append(diversity)
    
    # 如果不同批次的多样性标准差很小，可能存在模式崩塌
    diversity_std = np.std(results)
    
    if diversity_std < threshold:
        print("警告：检测到潜在的模式崩塌")
    
    return results
```

---

## 六、鲁棒性与泛化性实验

### 实验6.1：跨条件鲁棒性

**测试场景**：
```python
robustness_tests = {
    # 1. 条件样本质量变化
    'noise_levels': [0, 0.05, 0.1, 0.15, 0.2],
    
    # 2. 条件样本数量变化（已在实验1覆盖）
    'n_conditions': [2, 3, 5, 10],
    
    # 3. 条件样本异质性
    'heterogeneity': ['homogeneous', 'moderate', 'heterogeneous'],
    
    # 4. 不同严重程度的PD样本
    'severity': ['early_stage', 'moderate', 'advanced']
}

for noise_level in robustness_tests['noise_levels']:
    # 对条件样本添加噪声
    noisy_conditions = add_noise(condition_samples, noise_level)
    
    # 生成并评估
    generated = model.generate(noisy_conditions)
    quality = evaluate_quality(generated)
    
    # 期望：噪声<10%时，质量下降<5%
```

### 实验6.2：跨数据集零样本生成

**目标**：测试模型在完全未见数据集上的表现

```python
# 在Sande数据集上训练
model_sande = train(sande_dataset)

# 零样本测试：使用UNM的真实PD样本作为条件
unm_conditions = sample(unm_pd_dataset, n=5)
generated_unm = model_sande.generate(unm_conditions)

# 在UNM测试集上评估
unm_test_performance = evaluate(generated_unm, unm_test_set)

# 对比：在UNM上微调后的性能
model_finetuned = finetune(model_sande, unm_pd_dataset[:10])
generated_finetuned = model_finetuned.generate(unm_conditions)
finetuned_performance = evaluate(generated_finetuned, unm_test_set)
```

---

## 七、可解释性与特征分析实验

### 实验7.1：显著性特征提取

```python
def analyze_feature_importance(model, real_pd, generated_pd):
    """分析哪些特征在生成过程中被增强"""
    
    features_to_analyze = [
        'beta_power_motor',
        'theta_power_frontal',
        'alpha_power_occipital',
        'beta_gamma_pac',
        'inter_hemispheric_coherence',
        'burst_duration',
        'burst_rate'
    ]
    
    results = {}
    for feature in features_to_analyze:
        real_values = extract_feature(real_pd, feature)
        gen_values = extract_feature(generated_pd, feature)
        
        # 计算增强倍数
        enhancement_ratio = gen_values.mean() / real_values.mean()
        
        # 统计显著性
        t_stat, p_value = stats.ttest_ind(gen_values, real_values)
        
        # 效应量（Cohen's d）
        cohens_d = (gen_values.mean() - real_values.mean()) / \
                   np.sqrt((gen_values.std()**2 + real_values.std()**2) / 2)
        
        results[feature] = {
            'enhancement_ratio': enhancement_ratio,
            'p_value': p_value,
            'effect_size': cohens_d
        }
    
    return results
```

### 实验7.2：t-SNE/UMAP可视化

```python
from sklearn.manifold import TSNE
import umap

def visualize_feature_space(real_pd, generated_pd, real_healthy):
    """可视化特征空间分布"""
    
    # 提取高维特征
    features_real_pd = extract_features(real_pd)
    features_gen_pd = extract_features(generated_pd)
    features_healthy = extract_features(real_healthy)
    
    # 合并数据
    all_features = np.vstack([features_real_pd, features_gen_pd, features_healthy])
    labels = ['Real PD']*len(real_pd) + \
             ['Generated PD']*len(generated_pd) + \
             ['Healthy']*len(real_healthy)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    embedded = tsne.fit_transform(all_features)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    for label in ['Real PD', 'Generated PD', 'Healthy']:
        mask = [l == label for l in labels]
        plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                   label=label, alpha=0.6)
    
    # 期望结果：
    # 1. Generated PD应该紧密围绕Real PD
    # 2. Generated PD与Healthy应该有明显分离
    # 3. Generated PD的分布略微更紧凑（特征更一致）
```

---

## 八、消融实验

### 实验8.1：约束条件的作用

**目标**：验证各个生理约束的必要性

```python
ablation_configs = {
    'full_model': {
        'spectral_constraint': True,
        'pac_constraint': True,
        'spatial_constraint': True,
        'contrastive_loss': True
    },
    'no_spectral': {
        'spectral_constraint': False,
        'pac_constraint': True,
        'spatial_constraint': True,
        'contrastive_loss': True
    },
    'no_pac': {
        'spectral_constraint': True,
        'pac_constraint': False,
        'spatial_constraint': True,
        'contrastive_loss': True
    },
    # ... 其他配置
    'baseline': {
        'spectral_constraint': False,
        'pac_constraint': False,
        'spatial_constraint': False,
        'contrastive_loss': False
    }
}

for config_name, config in ablation_configs.items():
    model = train_model(config)
    generated = model.generate(n_samples=200)
    
    results[config_name] = {
        'fid': compute_fid(generated, real_pd),
        'downstream_acc': train_and_test_classifier(generated),
        'beta_enhancement': compute_beta_ratio(generated),
        'clinical_score': expert_evaluate(generated)
    }
```

---

## 九、长期稳定性实验

### 实验9.1：训练稳定性

```python
def evaluate_training_stability(n_runs=5):
    """评估训练的稳定性和可重复性"""
    
    results = []
    for run in range(n_runs):
        # 使用不同随机种子
        set_seed(run * 42)
        
        # 训练模型
        model = train_from_scratch(train_data, epochs=200)
        
        # 评估
        generated = model.generate(n_samples=500)
        metrics = evaluate_all_metrics(generated)
        
        results.append(metrics)
    
    # 计算各指标的均值和标准差
    mean_metrics = {k: np.mean([r[k] for r in results]) 
                   for k in results[0].keys()}
    std_metrics = {k: np.std([r[k] for r in results]) 
                  for k in results[0].keys()}
    
    # 变异系数 (CV = std/mean)
    cv = {k: std_metrics[k] / mean_metrics[k] 
          for k in mean_metrics.keys()}
    
    # 期望：主要指标的CV < 5%
    return mean_metrics, std_metrics, cv
```

---

## 十、实验报告结构建议

### 10.1 结果呈现

**表1：最少样本量实验结果**
| 输入样本数 | FID ↓ | IS ↑ | 下游准确率 | 临床评分 |
|----------|-------|------|----------|---------|
| 1 | 85.3±12.1 | 2.1±0.3 | 68.5±4.2 | 2.8±0.5 |
| 2 | 52.7±8.4 | 3.2±0.4 | 76.3±3.1 | 3.5±0.4 |
| **3** | **35.2±5.2** | **4.1±0.3** | **82.1±2.5** | **4.1±0.3** |
| 5 | 28.1±4.1 | 4.5±0.2 | 85.2±2.1 | 4.3±0.2 |

**结论**：最少需要3个输入样本即可生成有效的PD模拟数据

**图1：学习曲线与饱和度分析**
- X轴：训练样本量 (10-300)
- Y轴：各项性能指标
- 显示饱和点和理论上限

**表2：消融实验结果**
| 配置 | FID | Beta增强 | PAC强度 | 下游准确率 |
|-----|-----|---------|---------|-----------|
| 完整模型 | 18.5 | 2.1x | 0.15 | 87.3% |
| 无频谱约束 | 32.1 | 1.3x | 0.12 | 81.2% |
| 无PAC约束 | 24.7 | 1.9x | 0.08 | 83.5% |

### 10.2 统计分析

所有实验重复至少5次，报告均值±标准差。使用配对t检验比较不同配置，Bonferroni校正多重比较。

---

## 十一、实验时间线估算

| 实验阶段 | 预计时间 | 备注 |
|---------|---------|------|
| 最少样本量实验 | 1-2周 | 可并行多个配置 |
| 学习曲线实验 | 2-3周 | 需要训练多个模型 |
| 下游任务评估 | 1-2周 | 可复用生成的样本 |
| 临床评估 | 3-4周 | 需要协调专家时间 |
| 鲁棒性测试 | 1周 | |
| 可解释性分析 | 1周 | |
| 消融实验 | 2周 | |
| 总计 | **11-15周** | 约3-4个月 |

---

## 十二、关键成功指标（KPI）

定义实验成功的标准：

1. **生成质量**：FID < 20（相比真实PD数据）
2. **临床有效性**：专家盲评准确率 > 80%
3. **下游任务提升**：分类准确率提升 > 5%
4. **跨数据集泛化**：跨数据集准确率提升 > 3%
5. **特征增强**：Beta功率增强 > 1.5x，PAC强度 > 0.12
6. **多样性**：覆盖率 > 85%，成对距离标准差 > 阈值
7. **统计有效性**：与真实PD无显著差异（p>0.05），与健康对照有显著差异（p<0.01）

如果上述7项中有6项达标，认为模型成功。

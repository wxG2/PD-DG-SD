# 额外评估实验：全面评估Diffusion生成PD脑电的质量

## 除了基础指标外的补充实验

### 一、神经生理学有效性验证实验

#### 1.1 事件相关电位(ERP)一致性
**目标**: 验证生成数据是否保持了PD患者的ERP特征

```python
def validate_erp_consistency(generated, real_pd, paradigm='oddball'):
    """
    分析事件相关电位的一致性
    """
    # 提取ERP成分
    p300_gen = extract_p300_component(generated, paradigm)
    p300_real = extract_p300_component(real_pd, paradigm)
    
    # 比较关键特征
    metrics = {
        'latency_diff': abs(p300_gen.latency - p300_real.latency),
        'amplitude_diff': abs(p300_gen.amplitude - p300_real.amplitude),
        'waveform_correlation': correlate(p300_gen.waveform, p300_real.waveform)
    }
    
    # PD患者通常有P300潜伏期延长和振幅降低
    return metrics
```

**判定标准**:
- P300潜伏期差异 < 20ms
- 波形相关系数 > 0.7
- 振幅比例在0.8-1.2之间

#### 1.2 微观状态(Microstate)分析
**目标**: 验证脑电微观状态的拓扑特征

```python
def microstate_analysis(generated, real_pd):
    """
    分析EEG微观状态
    """
    # 识别4个标准微观状态 (A, B, C, D)
    ms_gen = identify_microstates(generated, n_states=4)
    ms_real = identify_microstates(real_pd, n_states=4)
    
    # 比较微观状态参数
    metrics = {
        'duration': compare_duration(ms_gen, ms_real),
        'occurrence': compare_occurrence(ms_gen, ms_real),
        'coverage': compare_coverage(ms_gen, ms_real),
        'transition_prob': compare_transitions(ms_gen, ms_real)
    }
    
    return metrics
```

**期望结果**: PD患者通常显示微观状态C和D的持续时间增加

---

### 二、时间动态特性验证

#### 2.1 长程时间相关性(DFA)
**目标**: 验证生成数据的长程相关性特征

```python
def detrended_fluctuation_analysis(generated, real_pd):
    """
    去趋势波动分析
    """
    # 计算标度指数α
    alpha_gen = compute_dfa_exponent(generated)
    alpha_real = compute_dfa_exponent(real_pd)
    
    # PD患者的α值通常偏离健康人(α≈1.0)
    # 检查生成数据是否保持这种偏离
    
    results = {
        'alpha_generated': alpha_gen,
        'alpha_real': alpha_real,
        'difference': abs(alpha_gen - alpha_real),
        'maintains_pd_pattern': abs(alpha_gen - 1.0) > 0.1
    }
    
    return results
```

#### 2.2 样本熵与多尺度熵
**目标**: 评估信号复杂度

```python
def complexity_analysis(generated, real_pd, real_healthy):
    """
    多尺度复杂度分析
    """
    scales = range(1, 21)
    
    # 计算多尺度熵
    mse_gen = multiscale_entropy(generated, scales)
    mse_pd = multiscale_entropy(real_pd, scales)
    mse_hc = multiscale_entropy(real_healthy, scales)
    
    # PD患者通常显示熵值降低（复杂度降低）
    results = {
        'mse_curve_similarity': correlate(mse_gen, mse_pd),
        'complexity_reduction': (mse_hc.mean() - mse_gen.mean()) / mse_hc.mean(),
        'scale_dependency': analyze_scale_dependency(mse_gen, mse_pd)
    }
    
    return results
```

---

### 三、脑网络拓扑验证

#### 3.1 图论网络指标
**目标**: 验证功能连接网络的拓扑特性

```python
def graph_theoretical_analysis(generated, real_pd):
    """
    基于图论的脑网络分析
    """
    # 构建功能连接网络
    network_gen = construct_functional_network(generated, method='plv')
    network_real = construct_functional_network(real_pd, method='plv')
    
    # 计算图论指标
    metrics = {
        # 全局指标
        'clustering_coef': nx.average_clustering(network_gen),
        'path_length': nx.average_shortest_path_length(network_gen),
        'global_efficiency': nx.global_efficiency(network_gen),
        'modularity': community.modularity(network_gen),
        
        # 局部指标
        'hub_nodes': identify_hubs(network_gen),
        'degree_distribution': nx.degree_histogram(network_gen),
        
        # 小世界属性
        'small_worldness': compute_small_worldness(network_gen)
    }
    
    # PD患者通常显示网络效率降低、模块化增加
    return metrics
```

#### 3.2 跨频段耦合网络
**目标**: 分析不同频段间的交互

```python
def cross_frequency_coupling_network(generated, real_pd):
    """
    跨频段耦合网络分析
    """
    freq_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 80)
    }
    
    # 计算所有频段对之间的耦合
    cfc_gen = compute_all_cfc_pairs(generated, freq_bands)
    cfc_real = compute_all_cfc_pairs(real_pd, freq_bands)
    
    # 特别关注theta-beta和beta-gamma耦合（PD的关键特征）
    key_couplings = {
        'theta_beta': compare_coupling(cfc_gen['theta-beta'], cfc_real['theta-beta']),
        'beta_gamma': compare_coupling(cfc_gen['beta-gamma'], cfc_real['beta-gamma']),
        'alpha_beta': compare_coupling(cfc_gen['alpha-beta'], cfc_real['alpha-beta'])
    }
    
    return key_couplings
```

---

### 四、鲁棒性与压力测试

#### 4.1 噪声鲁棒性测试
**目标**: 测试生成样本在加噪后的稳定性

```python
def noise_robustness_test(generated, noise_levels=[0.01, 0.05, 0.1, 0.2, 0.5]):
    """
    噪声鲁棒性测试
    """
    results = {}
    
    for noise_level in noise_levels:
        # 添加高斯白噪声
        noisy_generated = add_gaussian_noise(generated, noise_level)
        
        # 重新评估PD特征
        beta_power = compute_band_power(noisy_generated, (12, 30))
        pac_strength = compute_pac(noisy_generated)
        
        # 使用预训练分类器测试
        classification_acc = classify_pd(noisy_generated)
        
        results[noise_level] = {
            'beta_retained': beta_power / compute_band_power(generated, (12, 30)),
            'pac_retained': pac_strength / compute_pac(generated),
            'classification_retained': classification_acc
        }
    
    # 期望：在SNR>10dB时，特征保持率>80%
    return results
```

#### 4.2 时间扰动鲁棒性
**目标**: 测试对时间操作的鲁棒性

```python
def temporal_robustness_test(generated):
    """
    时间扰动鲁棒性测试
    """
    perturbations = {
        'time_shift': lambda x: time_shift(x, shift_range=(-0.5, 0.5)),
        'time_warp': lambda x: time_warping(x, warp_factor=(0.8, 1.2)),
        'segment_shuffle': lambda x: shuffle_segments(x, n_segments=5),
        'downsample_upsample': lambda x: resample(resample(x, 0.5), 2.0)
    }
    
    results = {}
    for pert_name, pert_func in perturbations.items():
        perturbed = pert_func(generated)
        
        # 评估特征保持
        feature_retention = compute_feature_similarity(perturbed, generated)
        
        results[pert_name] = {
            'feature_retention': feature_retention,
            'still_classifiable': classify_pd(perturbed) > 0.7
        }
    
    return results
```

---

### 五、生成条件敏感性分析

#### 5.1 条件插值实验
**目标**: 理解条件空间的连续性

```python
def condition_interpolation_experiment(model, pd_sample_a, pd_sample_b):
    """
    在两个PD样本之间进行条件插值
    """
    alphas = np.linspace(0, 1, 11)  # 11个插值点
    
    interpolated_generations = []
    for alpha in alphas:
        # 插值条件
        condition = alpha * encode(pd_sample_a) + (1-alpha) * encode(pd_sample_b)
        
        # 生成
        generated = model.generate(condition=condition)
        
        # 分析特征变化
        features = extract_features(generated)
        interpolated_generations.append(features)
    
    # 分析：特征应该平滑变化
    smoothness = analyze_smoothness(interpolated_generations)
    
    return {
        'interpolations': interpolated_generations,
        'smoothness_score': smoothness,
        'is_continuous': smoothness > 0.8
    }
```

#### 5.2 条件极端值测试
**目标**: 测试模型在极端条件下的表现

```python
def extreme_condition_test(model, pd_dataset):
    """
    测试极端条件输入
    """
    # 1. 轻度PD样本作为条件
    mild_pd = select_mild_cases(pd_dataset, severity='mild')
    gen_mild = model.generate(condition=mild_pd)
    
    # 2. 重度PD样本作为条件
    severe_pd = select_severe_cases(pd_dataset, severity='severe')
    gen_severe = model.generate(condition=severe_pd)
    
    # 3. 混合严重程度
    mixed = mild_pd[:2] + severe_pd[:2]
    gen_mixed = model.generate(condition=mixed)
    
    # 分析：生成的样本严重程度应该有梯度
    severity_scores = {
        'mild_condition': estimate_severity(gen_mild),
        'severe_condition': estimate_severity(gen_severe),
        'mixed_condition': estimate_severity(gen_mixed)
    }
    
    # 期望：mild < mixed < severe
    return severity_scores
```

---

### 六、生物学合理性验证

#### 6.1 生理可行性检查
**目标**: 确保生成的信号在生理上是可能的

```python
def physiological_plausibility_check(generated):
    """
    生理可行性检查清单
    """
    checks = {
        # 1. 振幅范围检查
        'amplitude_range': check_amplitude_range(generated, valid_range=(-100, 100)),
        
        # 2. 频率内容检查
        'frequency_content': check_frequency_range(generated, max_freq=100),
        
        # 3. 梯度检查（避免不自然的跳跃）
        'gradient_check': check_temporal_gradient(generated, max_gradient=50),
        
        # 4. 伪迹检查
        'artifact_free': check_artifacts(generated, types=['blink', 'muscle', 'electrode']),
        
        # 5. 空间一致性（相邻电极应该相关）
        'spatial_consistency': check_spatial_coherence(generated, min_coherence=0.3),
        
        # 6. 频带功率比例合理性
        'power_ratio_plausible': check_power_ratios(generated),
        
        # 7. 对称性检查（左右半球）
        'hemispheric_symmetry': check_hemispheric_symmetry(generated)
    }
    
    plausibility_score = sum(checks.values()) / len(checks)
    
    return {
        'individual_checks': checks,
        'overall_plausibility': plausibility_score,
        'is_plausible': plausibility_score > 0.85
    }
```

#### 6.2 源空间投影验证
**目标**: 验证源定位的合理性

```python
def source_localization_validation(generated, real_pd):
    """
    源定位验证
    """
    # 执行源定位
    sources_gen = inverse_solution(generated, method='sLORETA')
    sources_real = inverse_solution(real_pd, method='sLORETA')
    
    # PD相关的关键脑区
    roi_pd = ['motor_cortex', 'supplementary_motor_area', 
              'basal_ganglia', 'thalamus']
    
    # 检查关键脑区的激活
    activation_comparison = {}
    for roi in roi_pd:
        activation_comparison[roi] = {
            'generated': get_roi_activation(sources_gen, roi),
            'real': get_roi_activation(sources_real, roi),
            'similarity': spatial_correlation(
                sources_gen[roi], sources_real[roi]
            )
        }
    
    return activation_comparison
```

---

### 七、增量学习与在线适应性测试

#### 7.1 增量学习能力
**目标**: 测试模型能否从少量新数据中学习

```python
def incremental_learning_test(pretrained_model, new_pd_samples):
    """
    增量学习测试
    """
    # 基线性能
    baseline_quality = evaluate(pretrained_model.generate(n=100))
    
    # 使用少量新样本微调
    for n_new_samples in [1, 3, 5, 10]:
        # 微调
        finetuned = incremental_finetune(
            pretrained_model, 
            new_pd_samples[:n_new_samples],
            epochs=10
        )
        
        # 评估
        new_quality = evaluate(finetuned.generate(n=100))
        
        improvement = (new_quality - baseline_quality) / baseline_quality
        
        print(f"使用{n_new_samples}个新样本: 性能提升 {improvement:.2%}")
```

#### 7.2 域适应能力
**目标**: 测试跨数据集适应能力

```python
def domain_adaptation_test(model_trained_on_sande, unm_data):
    """
    域适应测试
    """
    # 零样本性能
    zero_shot_quality = evaluate_on_target_domain(
        model_trained_on_sande, 
        unm_data
    )
    
    # 少样本适应（3-shot, 5-shot, 10-shot）
    few_shot_results = {}
    for n_shots in [3, 5, 10, 20]:
        adapted_model = domain_adapt(
            model_trained_on_sande,
            unm_data[:n_shots],
            method='fine_tuning'
        )
        
        few_shot_results[n_shots] = evaluate_on_target_domain(
            adapted_model,
            unm_data
        )
    
    # 可视化适应曲线
    plot_adaptation_curve(zero_shot_quality, few_shot_results)
```

---

### 八、可控生成能力测试

#### 8.1 条件控制精度
**目标**: 测试能否精确控制生成特征

```python
def controllable_generation_test(model):
    """
    可控生成测试
    """
    # 测试不同的控制维度
    control_dimensions = {
        'beta_power': [1.0, 1.5, 2.0, 2.5, 3.0],  # 相对于基线的倍数
        'theta_alpha_ratio': [0.8, 1.0, 1.2, 1.5, 2.0],
        'pac_strength': [0.05, 0.10, 0.15, 0.20, 0.25]
    }
    
    results = {}
    for dimension, target_values in control_dimensions.items():
        dimension_results = []
        
        for target in target_values:
            # 生成时指定目标值
            generated = model.generate(
                n_samples=50,
                control={dimension: target}
            )
            
            # 测量实际值
            actual = measure_feature(generated, dimension)
            
            # 计算控制误差
            error = abs(actual - target) / target
            
            dimension_results.append({
                'target': target,
                'actual': actual,
                'error': error
            })
        
        results[dimension] = dimension_results
    
    # 期望：控制误差 < 10%
    return results
```

#### 8.2 独立属性控制
**目标**: 测试是否能独立控制不同属性

```python
def independent_control_test(model):
    """
    独立属性控制测试
    """
    # 同时控制多个属性
    control_configs = [
        {'beta': 2.0, 'theta': 1.0},  # 只增强beta
        {'beta': 1.0, 'theta': 1.5},  # 只增强theta
        {'beta': 2.0, 'theta': 1.5},  # 同时增强
    ]
    
    for config in control_configs:
        generated = model.generate(n_samples=100, control=config)
        
        # 验证控制独立性
        measured_beta = measure_beta_power(generated)
        measured_theta = measure_theta_power(generated)
        
        beta_correct = abs(measured_beta - config['beta']) < 0.1
        theta_correct = abs(measured_theta - config['theta']) < 0.1
        
        # 两个属性应该都达到目标
        assert beta_correct and theta_correct
```

---

### 九、实验评估总结表

| 实验类别 | 具体实验 | 关键指标 | 成功标准 | 优先级 |
|---------|---------|---------|---------|--------|
| 神经生理 | ERP一致性 | P300潜伏期/振幅 | 差异<20ms | 高 |
| 神经生理 | 微观状态 | 持续时间/转换概率 | 相关性>0.7 | 中 |
| 时间动态 | DFA | 标度指数α | \|α_gen-α_real\|<0.1 | 高 |
| 时间动态 | 多尺度熵 | MSE曲线 | 相关性>0.8 | 中 |
| 脑网络 | 图论分析 | 聚类系数/路径长度 | 相对误差<15% | 高 |
| 脑网络 | 跨频段耦合 | CFC强度 | theta-beta保持 | 高 |
| 鲁棒性 | 噪声测试 | SNR=10dB特征保持 | >80% | 高 |
| 鲁棒性 | 时间扰动 | 特征保持率 | >75% | 中 |
| 条件敏感性 | 插值连续性 | 平滑度得分 | >0.8 | 中 |
| 条件敏感性 | 极端值测试 | 严重程度梯度 | 单调性 | 低 |
| 生物合理性 | 可行性检查 | 7项检查 | >85%通过 | 高 |
| 生物合理性 | 源定位 | ROI激活相似度 | >0.6 | 中 |
| 适应性 | 增量学习 | 3-shot提升 | >5% | 低 |
| 适应性 | 域适应 | 跨数据集性能 | >70% | 高 |
| 可控性 | 控制精度 | 目标误差 | <10% | 中 |
| 可控性 | 独立控制 | 独立性验证 | 通过 | 低 |

---

### 十、建议的实验优先级排序

**第一阶段（必须完成）**：
1. 最少样本量实验
2. 学习曲线与质量上限
3. 基础生理有效性（Beta、PAC）
4. 下游任务性能
5. 跨数据集泛化

**第二阶段（强烈推荐）**：
6. 临床专家盲评
7. 多样性与模式崩塌
8. DFA长程相关性
9. 图论网络分析
10. 噪声鲁棒性

**第三阶段（锦上添花）**：
11. 微观状态分析
12. 多尺度熵
13. 源定位验证
14. 条件控制能力
15. 域适应测试

---

### 十一、实验结果展示建议

#### 推荐的图表类型：

1. **箱线图**: 不同配置下的性能分布
2. **学习曲线**: 训练样本量 vs 性能指标
3. **雷达图**: 多维度性能对比
4. **热图**: 混淆矩阵、相关性矩阵
5. **t-SNE/UMAP**: 特征空间可视化
6. **拓扑图**: 脑网络连接模式
7. **时频图**: 时频特征对比
8. **瀑布图**: 多个样本的频谱叠加

#### 统计检验要求：

- 所有比较使用配对/独立样本t检验
- 多重比较使用Bonferroni或FDR校正
- 报告效应量（Cohen's d）
- 提供95%置信区间
- 进行功效分析确保样本量充足

---

这套完整的评估体系可以从多个维度全面评估Diffusion模型生成PD脑电数据的质量，不仅关注生成质量本身，还关注临床相关性、神经生理学有效性和实用性。

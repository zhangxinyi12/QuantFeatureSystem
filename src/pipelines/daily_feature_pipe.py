def daily_feature_pipeline(symbols):
    # 0. 初始化监控
    resource_monitor = ResourceMonitor()
    resource_monitor.start()
    
    try:
        # 1. 数据获取
        db_conn = SafeDatabaseConnector()
        raw_data = db_conn.get_batch_data(symbols, Config.START_DATE, Config.END_DATE)
        
        # 2. 预处理
        cleaner = DataCleaner()
        cleaned_data = (
            cleaner.handle_missing(raw_data)
            .handle_outliers()
            .apply_calendar_adjustments()
        )
        
        # 3. 特征工程
        feature_maker = FeatureFactory(Config.CALC_TIMESTAMP)
        features = (
            feature_maker.add_basic_features(cleaned_data)
            .add_advanced_features()
            .add_market_features()
        )
        
        # 4. 验证与解释
        validator = FeatureValidator(Config.CALC_TIMESTAMP)
        validator.check_all_features(features)
        
        explainer = FeatureExplainer()
        feature_docs = explainer.generate_docs(features)
        
        # 5. 高效存储
        store = FeatureStore(partition_strategy='monthly')
        store.save_features(features, version='v1.0')
        
        # 6. 漂移检测
        drift_report = DriftDetector(Config.BASELINE_DATA).check_all(features)
        
        return {
            'status': 'success',
            'features': features,
            'docs': feature_docs,
            'drift_report': drift_report
        }
    
    except Exception as e:
        resource_monitor.alert(f"管道失败: {str(e)}")
        return {'status': 'error', 'message': str(e)}
    
    finally:
        resource_monitor.stop()
        resource_monitor.generate_report()

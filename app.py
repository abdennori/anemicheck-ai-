# ========== PAGE: ACCUEIL ==========
if st.session_state.page == "home":
    # Header avec badges
    if logo:
        st.markdown(f"""
        <div class="header">
            <div class="header-left">
                <img src="data:image/png;base64,{logo}" style="height:42px;">
                <div>
                    <h1>{t('app_title')}</h1>
                    <div class="subtitle">{t('app_subtitle')}</div>
                </div>
            </div>
            <div class="header-badges">
                <div class="header-badge">🔒 <strong>{t('badge_private')}</strong> {t('badge_private_desc')}</div>
                <div class="header-badge">⏰ <strong>{t('badge_available')}</strong> {t('badge_available_desc')}</div>
                <div class="header-badge">📞 <strong>{t('badge_call')}</strong> {t('badge_call_desc')}</div>
                <div class="header-badge">🆘 <strong>{t('badge_support')}</strong> {t('badge_support_desc')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="header">
            <div class="header-left">
                <div style="font-size:36px;">🩸</div>
                <div>
                    <h1>{t('app_title')}</h1>
                    <div class="subtitle">{t('app_subtitle')}</div>
                </div>
            </div>
            <div class="header-badges">
                <div class="header-badge">🔒 <strong>{t('badge_private')}</strong> {t('badge_private_desc')}</div>
                <div class="header-badge">⏰ <strong>{t('badge_available')}</strong> {t('badge_available_desc')}</div>
                <div class="header-badge">📞 <strong>{t('badge_call')}</strong> {t('badge_call_desc')}</div>
                <div class="header-badge">🆘 <strong>{t('badge_support')}</strong> {t('badge_support_desc')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown(f"""
    <div class="hero">
        <span class="icon">🩺</span>
        <h1>{t('hero_title')}</h1>
        <p>{t('hero_desc')}</p>
        <div style="margin-top:10px;">
            <span class="hero-badge">✅ {t('hero_badge')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== ZONE UPLOAD (AJOUTÉE ICI) ==========
    st.markdown(f"""
    <div class="upload-card">
        <div class="icon">📸</div>
        <h3>{t('upload_title')}</h3>
        <p>{t('upload_desc')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        option = st.radio(
            t("upload_method_file"),
            [t("upload_method_file"), t("upload_method_camera")],
            horizontal=True,
            label_visibility="collapsed",
            key="home_upload_method"
        )
    with col2:
        if option == t("upload_method_file"):
            uploaded = st.file_uploader(
                t("upload_file_label"),
                type=["jpg","png","jpeg"],
                label_visibility="collapsed",
                key="home_file_uploader"
            )
        else:
            uploaded = st.camera_input(
                t("upload_camera_label"),
                label_visibility="collapsed",
                key="home_camera_input"
            )
    
    # ========== TRAITEMENT DE L'IMAGE SUR LA PAGE D'ACCUEIL ==========
    if uploaded is not None:
        st.markdown(f"""
        <div class="preview-container">
            <h4 style="margin-top:0;">📷 {t('preview_title')}</h4>
        """, unsafe_allow_html=True)
        col_preview, _ = st.columns([1, 1])
        with col_preview:
            st.image(uploaded, use_container_width=True, caption=t('preview_caption'))
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button(t("analyze_btn"), use_container_width=True, key="home_analyze_btn"):
            with st.spinner(t("loading_models")):
                unet_model, unet_device, clf_model, clf_device = load_models()

            progress_bar = st.progress(0)
            for i in range(10):
                time.sleep(0.05)
                progress_bar.progress((i+1)*10)

            with st.spinner(t("analyzing")):
                img = np.array(Image.open(uploaded).convert('RGB'))
                img = cv2.flip(img, 1)

                transform_unet = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])
                tensor = transform_unet(img).unsqueeze(0).to(unet_device)
                with torch.no_grad():
                    raw_mask = torch.sigmoid(unet_model(tensor)).squeeze().cpu().numpy()
                    raw_mask = (raw_mask > 0.5).astype(np.uint8) * 255
                    raw_mask = cv2.resize(raw_mask, (img.shape[1], img.shape[0]))

                conj_enhanced, final_mask, bbox = extract_best_conjunctiva(img, raw_mask)

                result, confidence, raw_pred = predict_anemia(clf_model, conj_enhanced, clf_device)

                anemia_pct = raw_pred * 100
                non_pct = (1 - raw_pred) * 100

                progress_bar.empty()
                st.success(t("analysis_done"))

                # --- IMAGES ---
                st.markdown(f'<div class="section-title">{t("results_title")}</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**{t('result_original')}**")
                    st.image(img, use_container_width=True)
                with col2:
                    st.markdown(f"**{t('result_mask')}**")
                    st.image(final_mask, use_container_width=True, clamp=True)
                with col3:
                    st.markdown(f"**{t('result_conjunctiva')}**")
                    st.image(conj_enhanced, use_container_width=True)

                # --- MÉTRIQUES ---
                before = np.sum(raw_mask > 0) / 255
                after = np.sum(final_mask > 0) / 255
                reduction = ((before - after) / before * 100) if before > 0 else 0

                m1, m2 = st.columns(2)
                with m1:
                    st.metric(t("metric_surface"), f"{after:.0f} px²")
                with m2:
                    st.metric(t("metric_cleaning"), f"{reduction:.1f}%")

                # --- DIAGNOSTIC ---
                st.markdown(f'<div class="section-title">{t("diagnostic_title")}</div>', unsafe_allow_html=True)
                col_res, col_conf = st.columns(2)
                with col_res:
                    if result == "Anemic":
                        st.markdown(f"""
                        <div class="result-card positive">
                            <h2>{t('diagnostic_anemic')}</h2>
                            <div class="confidence">{t('diagnostic_confidence')} : <strong>{confidence:.1f}%</strong></div>
                            <div class="sub">{t('diagnostic_anemic_desc')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card negative">
                            <h2>{t('diagnostic_non_anemic')}</h2>
                            <div class="confidence">{t('diagnostic_confidence')} : <strong>{confidence:.1f}%</strong></div>
                            <div class="sub">{t('diagnostic_non_anemic_desc')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                with col_conf:
                    st.metric(t("diagnostic_confidence"), f"{confidence:.1f}%")
                    st.progress(int(confidence))

                # --- GRAPHIQUE ---
                st.markdown(f'<div class="section-title">{t("chart_title")}</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(8,5))
                cats = [t('chart_non'), t('chart_anemic')]
                vals = [non_pct, anemia_pct]
                colors = ['#10b981', '#dc2626']
                bars = ax.bar(cats, vals, color=colors, width=0.5, edgecolor='white', linewidth=2)
                ax.set_ylim(0,100)
                ax.set_ylabel('Pourcentage (%)')
                ax.set_title(t('chart_title'), fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, v+2, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=14)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)

                # --- HISTORIQUE ---
                entry = {
                    t("history_date"): datetime.now().strftime("%Y-%m-%d %H:%M"),
                    t("history_diagnostic"): result,
                    t("history_confidence"): f"{confidence:.1f}%",
                    t("history_prob"): f"{anemia_pct:.1f}%"
                }
                st.session_state.history.append(entry)
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop(0)

                if st.session_state.history:
                    st.markdown(f'<div class="section-title">{t("history_title")}</div>', unsafe_allow_html=True)
                    df = pd.DataFrame(st.session_state.history)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                # --- DÉTAILS TECHNIQUES ---
                with st.expander(t("tech_details")):
                    st.write(f"**{t('tech_model_seg')}:** U‑Net (ResNet34)")
                    st.write(f"**{t('tech_model_clf')}:** EfficientNet‑B3")
                    st.write(f"**{t('tech_device')}:** {'GPU' if clf_device.type == 'cuda' else 'CPU'}")
                    st.write(f"**{t('tech_sigmoid')}:** {raw_pred:.4f}")
                    st.write(f"**{t('tech_prob_anemic')}:** {anemia_pct:.1f}%")
                    st.write(f"**{t('tech_prob_non')}:** {non_pct:.1f}%")
                    st.write(f"**{t('tech_preprocess')}:** CLAHE + Filtrage + Netteté")
                    st.write(f"**{t('tech_decision')}:** {t('tech_decision_value')}")

                # --- AVERTISSEMENT ---
                st.markdown(f"""
                <div class="disclaimer">
                    <strong>{t('disclaimer')}</strong><br>
                    {t('disclaimer_text')}<br>
                    {t('disclaimer_consult')}
                </div>
                """, unsafe_allow_html=True)
    
    # Features Grid
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <div class="icon">⚡</div>
            <h4>""" + t('feature_1_title') + """</h4>
            <p>""" + t('feature_1_desc') + """</p>
        </div>
        <div class="feature-card">
            <div class="icon">📤</div>
            <h4>""" + t('feature_2_title') + """</h4>
            <p>""" + t('feature_2_desc') + """</p>
        </div>
        <div class="feature-card">
            <div class="icon">✅</div>
            <h4>""" + t('feature_3_title') + """</h4>
            <p>""" + t('feature_3_desc') + """</p>
        </div>
        <div class="feature-card">
            <div class="icon">🔒</div>
            <h4>""" + t('feature_4_title') + """</h4>
            <p>""" + t('feature_4_desc') + """</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works
    st.markdown(f"""
    <div class="how-section">
        <h3>{t('how_title')}</h3>
        <div class="how-steps">
            <div class="how-step">
                <div class="step-num">1</div>
                <h5>{t('how_step1')}</h5>
                <p>{t('how_step1_desc')}</p>
            </div>
            <div class="how-step">
                <div class="step-num">2</div>
                <h5>{t('how_step2')}</h5>
                <p>{t('how_step2_desc')}</p>
            </div>
            <div class="how-step">
                <div class="step-num">3</div>
                <h5>{t('how_step3')}</h5>
                <p>{t('how_step3_desc')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Trust badges
    st.markdown(f"""
    <div class="trust-section">
        <div class="trust-item"><span class="icon">🔒</span> {t('trust_1')}</div>
        <div class="trust-item"><span class="icon">❤️</span> {t('trust_2')}</div>
        <div class="trust-item"><span class="icon">⭐</span> {t('trust_3')}</div>
        <div class="trust-item"><span class="icon">📱</span> {t('version')}</div>
        <div class="trust-item"><span class="icon">🎯</span> {t('accuracy')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bottom info
    st.markdown(f"""
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:1.5rem 0;">
        <div class="feature-card">
            <div class="icon">⚡</div>
            <h4>{t('fast')}</h4>
            <p>{t('fast_desc')}</p>
        </div>
        <div class="feature-card">
            <div class="icon">🧠</div>
            <h4>{t('smart')}</h4>
            <p>{t('smart_desc')}</p>
        </div>
        <div class="feature-card">
            <div class="icon">🔒</div>
            <h4>{t('secure')}</h4>
            <p>{t('secure_desc')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

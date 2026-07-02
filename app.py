import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import base64
import os
import time
import pandas as pd
import io
from datetime import datetime
import random
from model_loader import load_unet_model, load_classifier_model

# ========== إعداد الصفحة ==========
st.set_page_config(
    page_title="AnemiCheck - Détection d'Anémie",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== الترجمة ==========
LANGUAGES = {
    "ar": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "الذكاء الاصطناعي لصحة الدم",
        "badge_private": "أمين وموثوق",
        "badge_private_desc": "100% خصوصيتك محمية",
        "badge_available": "خدمة متاحة",
        "badge_available_desc": "24/7 نحن هنا من أجلك دائماً",
        "badge_call": "مكالمة مجانية",
        "badge_call_desc": "+213 123 456 789",
        "badge_support": "مساعدة فورية",
        "badge_support_desc": "تواصل مع فريق الدعم",
        "hero_title": "طبيبك معك بأقل من دقيقتين",
        "hero_desc": "تحليل ذكي لصورة الدم لكشف الأنيميا بدقة وسرعة عالية",
        "hero_badge": "100% نتائج دقيقة وآمنة",
        "hero_cta": "ابدأ التحليل الآن",
        "ai_dev_badge": "تم تطويره بواسطة الذكاء الاصطناعي",
        "feature_1_title": "نتيجة سريعة",
        "feature_1_desc": "احصل على النتيجة فوراً مع تقرير مفصل",
        "feature_2_title": "رفع صورة الدم",
        "feature_2_desc": "قم برفع صورة واضحة لشريحة الدم (JPG, PNG)",
        "feature_3_title": "موثوق ومعتمد",
        "feature_3_desc": "هذا التطبيق يساعد في الكشف المبكر عن الأنيميا باستخدام الذكاء الاصطناعي",
        "feature_4_title": "أمان",
        "feature_4_desc": "بياناتك محمية بالكامل",
        "how_title": "كيف يعمل التطبيق؟",
        "how_step1": "رفع الصورة",
        "how_step1_desc": "ارفع صورة واضحة لشريحة الدم",
        "how_step2": "تحليل ذكي",
        "how_step2_desc": "يستخدم النموذج الذكاء الاصطناعي للتحليل",
        "how_step3": "النتيجة الفورية",
        "how_step3_desc": "احصل على التشخيص في ثوانٍ",
        "trust_1": "رعاية",
        "trust_2": "ثقة عالية",
        "trust_3": "سرعة",
        "trust_4": "ذكاء",
        "version": "الإصدار 1.0.0",
        "accuracy": "دقة تصل إلى 96%",
        "fast": "سريع",
        "fast_desc": "النتيجة في أقل من دقيقتين",
        "smart": "ذكي",
        "smart_desc": "تقنية ذكاء اصطناعي متقدمة",
        "secure": "أمان",
        "secure_desc": "بياناتك محمية بالكامل",
        "upload_title": "رفع صورة الدم",
        "upload_desc": "صورة واضحة لشريحة الدم للكشف عن الأنيميا",
        "upload_method_file": "📁 رفع صورة",
        "upload_method_camera": "📷 تصوير",
        "upload_file_label": "اختر صورة من جهازك",
        "upload_camera_label": "التقط صورة بالكاميرا",
        "preview_title": "📷 معاينة الصورة",
        "preview_caption": "الصورة المرفوعة",
        "analyze_btn": "🔍 تحليل الصورة",
        "loading_models": "⏳ جارٍ تحميل النماذج...",
        "analyzing": "⏳ جارٍ التحليل...",
        "analysis_done": "✅ تم التحليل بنجاح",
        "results_title": "📊 نتائج التحليل",
        "result_original": "🖼️ الصورة الأصلية",
        "result_mask": "🎭 القناع النهائي",
        "result_conjunctiva": "👁️ المنطقة المحسّنة",
        "metric_surface": "📐 المساحة المقطعة",
        "metric_cleaning": "🧼 التنظيف",
        "diagnostic_title": "🩺 التشخيص",
        "diagnostic_anemic": "🩸 أنيميا",
        "diagnostic_anemic_desc": "يوجد أنيميا",
        "diagnostic_non_anemic": "✅ لا يوجد أنيميا",
        "diagnostic_non_anemic_desc": "لا يوجد أنيميا",
        "diagnostic_confidence": "📊 مستوى الثقة",
        "chart_title": "📈 توزيع الاحتمالات",
        "chart_non": "غير مصاب",
        "chart_anemic": "مصاب",
        "history_title": "📋 النتائج السابقة",
        "history_date": "التاريخ",
        "history_diagnostic": "التشخيص",
        "history_confidence": "الثقة",
        "history_prob": "احتمال الأنيميا",
        "history_empty": "📭 لا توجد نتائج سابقة بعد",
        "history_empty_desc": "قم بإجراء أول تحليل لك الآن!",
        "history_stats_title": "📊 إحصائيات سريعة",
        "history_total": "إجمالي التحاليل",
        "history_positive": "إيجابي (مصاب)",
        "history_negative": "سلبي (غير مصاب)",
        "history_avg_conf": "متوسط الثقة",
        "history_chart_title": "📈 تطور الاحتمال مع مرور الوقت",
        "history_export": "📥 تصدير التقرير",
        "history_clear": "🗑️ مسح السجل",
        "tech_details": "📘 التفاصيل التقنية",
        "tech_model_seg": "نموذج التجزئة",
        "tech_model_clf": "نموذج التصنيف",
        "tech_device": "الجهاز المستخدم",
        "tech_sigmoid": "قيمة السيجمويد الخام",
        "tech_prob_anemic": "احتمال الأنيميا",
        "tech_prob_non": "احتمال عدم الإصابة",
        "tech_preprocess": "المعالجة المسبقة",
        "tech_decision": "القرار",
        "tech_decision_value": "مستقل تماماً، بدون تحيز",
        "disclaimer": "⚠️ تنويه طبي",
        "disclaimer_text": "هذه النتيجة صادرة عن نموذج ذكاء اصطناعي ولا تغني عن استشارة الطبيب المختص.",
        "disclaimer_consult": "استشر طبيباً للحصول على تشخيص دقيق.",
        "lang_selector": "🌐 اللغة",
        "sidebar_qr_title": "📱 وصول سريع",
        "sidebar_qr_desc": "امسح الكود لفتح التطبيق",
        "sidebar_health_title": "💡 نصائح صحية",
        "sidebar_health_1": "التغذية: ركز على الأطعمة الغنية بالحديد.",
        "sidebar_health_2": "فيتامين C: يحسن امتصاص الحديد.",
        "sidebar_health_3": "الترطيب: اشرب 1.5 لتر ماء يومياً.",
        "sidebar_health_4": "استشر الطبيب عند الشعور بالتعب الشديد.",
        "sidebar_doctor_title": "🩺 استشارة طبيب",
        "sidebar_doctor_desc": "احجز موعداً أو تحدث مع طبيب مباشرة.",
        "sidebar_doctor_btn": "📅 حجز استشارة",
        "sidebar_doctor_soon": "🔜 متاح في التحديث القادم",
        "sidebar_version": "AnemiCheck v2.0 • الذكاء الاصطناعي الطبي",
        "nav_home": "🏠 الرئيسية",
        "nav_blood_analysis": "🩸 تحليل صورة الدم",
        "nav_history": "📋 النتائج السابقة",
        "nav_disease_info": "📖 معلومات المرض",
        "nav_tips": "💡 نصائح صحية",
        "nav_about": "ℹ️ حول التطبيق",
        "disease_info_title": "📖 معلومات عن الأنيميا",
        "disease_info_desc": "الأنيميا (فقر الدم) هي حالة نقص في عدد كريات الدم الحمراء أو الهيموغلوبين، مما يؤدي إلى نقص الأكسجين في الجسم.",
        "disease_info_symptoms": "الأعراض الشائعة: تعب، شحوب، دوخة، ضيق في التنفس، برودة الأطراف.",
        "disease_info_causes": "الأسباب: نقص الحديد، نقص فيتامين B12، أمراض مزمنة، نزيف، أمراض وراثية.",
        "disease_info_prevention": "الوقاية: تناول غذاء متوازن غني بالحديد، فيتامين C، وحمض الفوليك، وممارسة الرياضة.",
        "disease_info_treatment": "العلاج: يعتمد على السبب، وقد يشمل مكملات الحديد، فيتامينات، أو علاجات أخرى حسب وصف الطبيب.",
        "about_title": "ℹ️ حول التطبيق",
        "about_desc": "AnemiCheck هو تطبيق ذكاء اصطناعي طبي يستخدم تقنيات التعلم العميق لتحليل صور العين (أو صور شرائح الدم) للكشف المبكر عن الأنيميا.",
        "about_mission": "مهمتنا: توفير أداة مساعدة سريعة ودقيقة للكشف المبكر عن الأنيميا، لدعم الأطباء والمستخدمين في متابعة صحتهم.",
        "about_team": "تم تطوير هذا التطبيق بواسطة فريق متخصص في الذكاء الاصطناعي والصحة الرقمية.",
        "about_contact": "للتواصل: support@anemicheck.com",
        "about_version": "الإصدار: 2.0.0",
        "about_tech": "التقنيات المستخدمة: PyTorch، Streamlit، OpenCV، U-Net، EfficientNet."
    },
    "fr": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "L'IA pour la santé sanguine",
        "badge_private": "Sécurisé et fiable",
        "badge_private_desc": "100% de confidentialité garantie",
        "badge_available": "Service disponible",
        "badge_available_desc": "24/7 nous sommes là pour vous",
        "badge_call": "Appel gratuit",
        "badge_call_desc": "+213 123 456 789",
        "badge_support": "Aide immédiate",
        "badge_support_desc": "Contactez notre équipe",
        "hero_title": "Votre médecin en moins de deux minutes",
        "hero_desc": "Analyse intelligente de l'image sanguine pour détecter l'anémie avec précision et rapidité",
        "hero_badge": "100% de résultats précis et sécurisés",
        "hero_cta": "Commencer l'analyse",
        "ai_dev_badge": "Développé avec l'intelligence artificielle",
        "feature_1_title": "Résultat rapide",
        "feature_1_desc": "Obtenez le résultat immédiatement avec un rapport détaillé",
        "feature_2_title": "Télécharger l'image sanguine",
        "feature_2_desc": "Téléchargez une image claire d'une lame de sang (JPG, PNG)",
        "feature_3_title": "Fiable et approuvé",
        "feature_3_desc": "Cette application aide à la détection précoce de l'anémie avec l'IA",
        "feature_4_title": "Sécurité",
        "feature_4_desc": "Vos données sont entièrement protégées",
        "how_title": "Comment ça marche ?",
        "how_step1": "Téléchargement",
        "how_step1_desc": "Téléchargez une image claire d'une lame de sang",
        "how_step2": "Analyse intelligente",
        "how_step2_desc": "Le modèle d'IA analyse l'image",
        "how_step3": "Résultat instantané",
        "how_step3_desc": "Obtenez le diagnostic en quelques secondes",
        "trust_1": "Soins",
        "trust_2": "Haute confiance",
        "trust_3": "Rapide",
        "trust_4": "Intelligent",
        "version": "Version 1.0.0",
        "accuracy": "Précision jusqu'à 96%",
        "fast": "Rapide",
        "fast_desc": "Résultat en moins de 2 minutes",
        "smart": "Intelligent",
        "smart_desc": "Technologie IA avancée",
        "secure": "Sécurisé",
        "secure_desc": "Vos données sont protégées",
        "upload_title": "Télécharger une image sanguine",
        "upload_desc": "Image claire d'une lame de sang pour la détection de l'anémie",
        "upload_method_file": "📁 Télécharger",
        "upload_method_camera": "📷 Appareil photo",
        "upload_file_label": "Choisissez une image depuis votre appareil",
        "upload_camera_label": "Prenez une photo avec la caméra",
        "preview_title": "📷 Aperçu de l'image",
        "preview_caption": "Image chargée",
        "analyze_btn": "🔍 Analyser l'image",
        "loading_models": "⏳ Chargement des modèles...",
        "analyzing": "⏳ Analyse en cours...",
        "analysis_done": "✅ Analyse terminée",
        "results_title": "📊 Résultats de l'analyse",
        "result_original": "🖼️ Image originale",
        "result_mask": "🎭 Masque final",
        "result_conjunctiva": "👁️ Zone améliorée",
        "metric_surface": "📐 Surface segmentée",
        "metric_cleaning": "🧼 Nettoyage",
        "diagnostic_title": "🩺 Diagnostic",
        "diagnostic_anemic": "🩸 Anémie",
        "diagnostic_anemic_desc": "Anémie détectée",
        "diagnostic_non_anemic": "✅ Non Anémie",
        "diagnostic_non_anemic_desc": "Pas d'anémie détectée",
        "diagnostic_confidence": "📊 Niveau de confiance",
        "chart_title": "📈 Distribution des probabilités",
        "chart_non": "Non Anémique",
        "chart_anemic": "Anémique",
        "history_title": "📋 Résultats précédents",
        "history_date": "Date",
        "history_diagnostic": "Diagnostic",
        "history_confidence": "Confiance",
        "history_prob": "Probabilité Anémie",
        "history_empty": "📭 Aucun résultat",
        "history_empty_desc": "Effectuez votre première analyse maintenant !",
        "history_stats_title": "📊 Statistiques rapides",
        "history_total": "Total analyses",
        "history_positive": "Positif (anémique)",
        "history_negative": "Négatif (non anémique)",
        "history_avg_conf": "Confiance moyenne",
        "history_chart_title": "📈 Évolution de la probabilité",
        "history_export": "📥 Exporter le rapport",
        "history_clear": "🗑️ Effacer l'historique",
        "tech_details": "📘 Détails techniques",
        "tech_model_seg": "Modèle de segmentation",
        "tech_model_clf": "Modèle de classification",
        "tech_device": "Appareil utilisé",
        "tech_sigmoid": "Valeur sigmoïde brute",
        "tech_prob_anemic": "Probabilité Anémie",
        "tech_prob_non": "Probabilité Non Anémie",
        "tech_preprocess": "Prétraitement",
        "tech_decision": "Décision",
        "tech_decision_value": "100% autonome, sans biais",
        "disclaimer": "⚠️ Avertissement médical",
        "disclaimer_text": "Ce résultat est généré par un modèle d'IA et ne remplace pas un avis médical.",
        "disclaimer_consult": "Consultez un professionnel de santé.",
        "lang_selector": "🌐 Langue",
        "sidebar_qr_title": "📱 Accès rapide",
        "sidebar_qr_desc": "Scannez pour ouvrir l'application",
        "sidebar_health_title": "💡 Conseils santé",
        "sidebar_health_1": "Alimentation : aliments riches en fer.",
        "sidebar_health_2": "Vitamine C : améliore l'absorption du fer.",
        "sidebar_health_3": "Hydratation : 1,5 L d'eau par jour.",
        "sidebar_health_4": "Consultez un médecin en cas de fatigue.",
        "sidebar_doctor_title": "🩺 Consultation médecin",
        "sidebar_doctor_desc": "Prenez rendez-vous ou discutez avec un médecin.",
        "sidebar_doctor_btn": "📅 Réserver",
        "sidebar_doctor_soon": "🔜 Bientôt disponible",
        "sidebar_version": "AnemiCheck v2.0 • IA médicale",
        "nav_home": "🏠 Accueil",
        "nav_blood_analysis": "🩸 Analyse de sang",
        "nav_history": "📋 Résultats précédents",
        "nav_disease_info": "📖 Info maladie",
        "nav_tips": "💡 Conseils santé",
        "nav_about": "ℹ️ À propos",
        "disease_info_title": "📖 À propos de l'anémie",
        "disease_info_desc": "L'anémie est une condition caractérisée par une diminution du nombre de globules rouges ou de l'hémoglobine, entraînant un manque d'oxygène dans le corps.",
        "disease_info_symptoms": "Symptômes courants : fatigue, pâleur, vertiges, essoufflement, extrémités froides.",
        "disease_info_causes": "Causes : carence en fer, carence en vitamine B12, maladies chroniques, saignements, maladies héréditaires.",
        "disease_info_prevention": "Prévention : une alimentation équilibrée riche en fer, vitamine C et acide folique, et pratiquer une activité physique.",
        "disease_info_treatment": "Traitement : dépend de la cause, peut inclure des suppléments de fer, des vitamines, ou d'autres traitements prescrits par un médecin.",
        "about_title": "ℹ️ À propos",
        "about_desc": "AnemiCheck est une application d'IA médicale utilisant le deep learning pour analyser des images de l'œil (ou de lames de sang) pour la détection précoce de l'anémie.",
        "about_mission": "Notre mission : fournir un outil d'assistance rapide et précis pour la détection précoce de l'anémie, afin d'aider les médecins et les utilisateurs à suivre leur santé.",
        "about_team": "Cette application a été développée par une équipe spécialisée en IA et santé numérique.",
        "about_contact": "Contact : support@anemicheck.com",
        "about_version": "Version : 2.0.0",
        "about_tech": "Technologies utilisées : PyTorch, Streamlit, OpenCV, U-Net, EfficientNet."
    },
    "en": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "AI for Blood Health",
        "badge_private": "Secure & Trusted",
        "badge_private_desc": "100% privacy protected",
        "badge_available": "Service Available",
        "badge_available_desc": "24/7 we are here for you",
        "badge_call": "Free Call",
        "badge_call_desc": "+213 123 456 789",
        "badge_support": "Instant Help",
        "badge_support_desc": "Chat with our support team",
        "hero_title": "Your doctor in less than two minutes",
        "hero_desc": "Intelligent analysis of blood images to detect anemia with high accuracy and speed",
        "hero_badge": "100% accurate and secure results",
        "hero_cta": "Start analysis now",
        "ai_dev_badge": "Built with artificial intelligence",
        "feature_1_title": "Quick Result",
        "feature_1_desc": "Get instant results with a detailed report",
        "feature_2_title": "Upload Blood Image",
        "feature_2_desc": "Upload a clear blood slide image (JPG, PNG)",
        "feature_3_title": "Reliable & Approved",
        "feature_3_desc": "This app helps in early detection of anemia using AI",
        "feature_4_title": "Security",
        "feature_4_desc": "Your data is fully protected",
        "how_title": "How it works?",
        "how_step1": "Upload",
        "how_step1_desc": "Upload a clear blood slide image",
        "how_step2": "Smart Analysis",
        "how_step2_desc": "AI model analyzes the image",
        "how_step3": "Instant Result",
        "how_step3_desc": "Get diagnosis in seconds",
        "trust_1": "Care",
        "trust_2": "High Trust",
        "trust_3": "Speed",
        "trust_4": "Intelligence",
        "version": "Version 1.0.0",
        "accuracy": "Accuracy up to 96%",
        "fast": "Fast",
        "fast_desc": "Result in less than 2 minutes",
        "smart": "Smart",
        "smart_desc": "Advanced AI technology",
        "secure": "Secure",
        "secure_desc": "Your data is protected",
        "upload_title": "Upload Blood Image",
        "upload_desc": "Clear blood slide image for anemia detection",
        "upload_method_file": "📁 Upload",
        "upload_method_camera": "📷 Camera",
        "upload_file_label": "Choose an image from your device",
        "upload_camera_label": "Take a photo with camera",
        "preview_title": "📷 Image Preview",
        "preview_caption": "Uploaded image",
        "analyze_btn": "🔍 Analyze Image",
        "loading_models": "⏳ Loading models...",
        "analyzing": "⏳ Analyzing...",
        "analysis_done": "✅ Analysis complete",
        "results_title": "📊 Analysis Results",
        "result_original": "🖼️ Original Image",
        "result_mask": "🎭 Final Mask",
        "result_conjunctiva": "👁️ Enhanced Area",
        "metric_surface": "📐 Segmented Area",
        "metric_cleaning": "🧼 Cleaning",
        "diagnostic_title": "🩺 Diagnosis",
        "diagnostic_anemic": "🩸 Anemia",
        "diagnostic_anemic_desc": "Anemia detected",
        "diagnostic_non_anemic": "✅ No Anemia",
        "diagnostic_non_anemic_desc": "No anemia detected",
        "diagnostic_confidence": "📊 Confidence Level",
        "chart_title": "📈 Probability Distribution",
        "chart_non": "Non Anemic",
        "chart_anemic": "Anemic",
        "history_title": "📋 Previous Results",
        "history_date": "Date",
        "history_diagnostic": "Diagnosis",
        "history_confidence": "Confidence",
        "history_prob": "Anemia Probability",
        "history_empty": "📭 No results yet",
        "history_empty_desc": "Perform your first analysis now!",
        "history_stats_title": "📊 Quick Stats",
        "history_total": "Total Analyses",
        "history_positive": "Positive (Anemic)",
        "history_negative": "Negative (Non-Anemic)",
        "history_avg_conf": "Avg Confidence",
        "history_chart_title": "📈 Probability Trend Over Time",
        "history_export": "📥 Export Report",
        "history_clear": "🗑️ Clear History",
        "tech_details": "📘 Technical Details",
        "tech_model_seg": "Segmentation Model",
        "tech_model_clf": "Classification Model",
        "tech_device": "Device Used",
        "tech_sigmoid": "Raw Sigmoid Value",
        "tech_prob_anemic": "Anemia Probability",
        "tech_prob_non": "Non-Anemia Probability",
        "tech_preprocess": "Preprocessing",
        "tech_decision": "Decision",
        "tech_decision_value": "100% autonomous, no bias",
        "disclaimer": "⚠️ Medical Disclaimer",
        "disclaimer_text": "This result is generated by an AI model and does not replace professional medical advice.",
        "disclaimer_consult": "Consult a healthcare professional.",
        "lang_selector": "🌐 Language",
        "sidebar_qr_title": "📱 Quick Access",
        "sidebar_qr_desc": "Scan to open the app",
        "sidebar_health_title": "💡 Health Tips",
        "sidebar_health_1": "Diet: focus on iron-rich foods.",
        "sidebar_health_2": "Vitamin C: improves iron absorption.",
        "sidebar_health_3": "Hydration: drink 1.5 L of water daily.",
        "sidebar_health_4": "Consult a doctor if you experience fatigue.",
        "sidebar_doctor_title": "🩺 Doctor Consultation",
        "sidebar_doctor_desc": "Book an appointment or chat with a doctor.",
        "sidebar_doctor_btn": "📅 Book Appointment",
        "sidebar_doctor_soon": "🔜 Coming soon",
        "sidebar_version": "AnemiCheck v2.0 • Medical AI",
        "nav_home": "🏠 Home",
        "nav_blood_analysis": "🩸 Blood Analysis",
        "nav_history": "📋 Previous Results",
        "nav_disease_info": "📖 Disease Info",
        "nav_tips": "💡 Health Tips",
        "nav_about": "ℹ️ About",
        "disease_info_title": "📖 About Anemia",
        "disease_info_desc": "Anemia is a condition characterized by a decrease in the number of red blood cells or hemoglobin, leading to a lack of oxygen in the body.",
        "disease_info_symptoms": "Common symptoms: fatigue, pallor, dizziness, shortness of breath, cold extremities.",
        "disease_info_causes": "Causes: iron deficiency, vitamin B12 deficiency, chronic diseases, bleeding, genetic disorders.",
        "disease_info_prevention": "Prevention: a balanced diet rich in iron, vitamin C, and folic acid, and regular exercise.",
        "disease_info_treatment": "Treatment: depends on the cause, may include iron supplements, vitamins, or other treatments prescribed by a doctor.",
        "about_title": "ℹ️ About",
        "about_desc": "AnemiCheck is a medical AI application using deep learning to analyze eye images (or blood slide images) for early detection of anemia.",
        "about_mission": "Our mission: provide a fast and accurate assistive tool for early anemia detection, to support doctors and users in monitoring their health.",
        "about_team": "This app was developed by a team specialized in AI and digital health.",
        "about_contact": "Contact: support@anemicheck.com",
        "about_version": "Version: 2.0.0",
        "about_tech": "Technologies used: PyTorch, Streamlit, OpenCV, U-Net, EfficientNet."
    }
}

def t(key):
    lang = st.session_state.get("language", "fr")
    return LANGUAGES.get(lang, LANGUAGES["fr"]).get(key, key)

# ========== CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Tajawal:wght@400;700;800&display=swap');
    
    * { font-family: 'Inter', 'Tajawal', sans-serif; }
    
    .stApp {
        background: linear-gradient(-45deg, #f4f7fb, #eaf1fb, #f1f5fb, #f4f7fb);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .header {
        background: linear-gradient(120deg, #1e3a8a 0%, #1d4ed8 55%, #2563eb 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 0 0 26px 26px;
        box-shadow: 0 10px 32px rgba(30,58,138,0.25);
        margin-bottom: 1.5rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        animation: slideDown 0.6s ease;
        gap: 12px;
    }
    @keyframes slideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 12px;
        order: 2;
    }
    .header-left h1 {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .header-left h1 span { color: #7dd3fc; }
    .header-left .subtitle {
        font-size: 12px;
        color: #dbeafe;
        font-weight: 500;
    }
    .header-badges {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        order: 1;
    }
    .header-ai-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 600;
        color: #ffffff;
        background: rgba(16,185,129,0.25);
        border: 1px solid rgba(16,185,129,0.5);
        padding: 5px 14px;
        border-radius: 30px;
    }
    .header-logo-badge {
        width: 48px;
        height: 48px;
        min-width: 48px;
        border-radius: 14px;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        overflow: hidden;
    }
    .header-logo-badge img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        padding: 4px;
    }
    .header-icon-btn {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: rgba(255,255,255,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-size: 15px;
        position: relative;
    }
    .header-icon-btn .dot {
        position: absolute;
        top: 4px;
        right: 4px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #f87171;
        border: 1.5px solid #1e3a8a;
    }
    .header-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        color: #334155;
        background: rgba(255,255,255,0.5);
        padding: 4px 12px;
        border-radius: 30px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .header-badge strong {
        color: #2563eb;
        font-weight: 700;
    }
    @media (max-width: 700px) {
        .header {
            flex-direction: column;
            align-items: stretch;
            padding: 0.8rem 1rem;
        }
        .header-left {
            order: 1;
            justify-content: center;
            flex-wrap: wrap;
        }
        .header-badges {
            order: 2;
            justify-content: center;
        }
        .header-left h1 { font-size: 20px; }
    }

    .top-badges {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 0 0 1.5rem;
    }
    @media (max-width: 768px) {
        .top-badges { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 480px) {
        .top-badges { grid-template-columns: 1fr; }
    }
    .top-badge-card {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 18px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 16px rgba(30,58,138,0.05);
        transition: all 0.3s ease;
        position: relative;
        min-height: 64px;
    }
    .top-badge-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(30,58,138,0.1);
    }
    .top-badge-card .tb-icon {
        width: 40px;
        height: 40px;
        min-width: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #F59E0B, #D97706);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 18px;
        box-shadow: 0 4px 10px rgba(245, 158, 11, 0.3);
    }
    .top-badge-card .tb-title {
        font-weight: 700;
        font-size: 13px;
        color: #0f172a;
    }
    .top-badge-card .tb-desc {
        font-size: 11px;
        color: #64748b;
    }

    .sidebar-glass {
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 24px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
        animation: fadeUp 0.8s ease;
    }
    .sidebar-glass h4 {
        color: #0f172a;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 15px;
    }
    .sidebar-glass p, .sidebar-glass li {
        color: #334155;
        font-size: 14px;
        line-height: 1.6;
    }
    .sidebar-glass ul {
        padding-left: 1.2rem;
    }
    .sidebar-glass .qr-container {
        display: flex;
        justify-content: center;
        margin: 0.8rem 0;
    }
    .sidebar-glass .qr-container img {
        border-radius: 16px;
        border: 1px solid rgba(245,158,11,0.15);
        background: white;
        padding: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        transition: transform 0.3s ease;
        max-width: 100%;
        height: auto;
    }
    .sidebar-glass .qr-container img:hover {
        transform: scale(1.05);
    }
    
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #2563eb 100%);
        border-radius: 32px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        min-height: 50vh;
        display: flex;
        align-items: center;
        box-shadow: 0 16px 40px rgba(30,58,138,0.25);
        animation: fadeUp 0.8s ease;
        transition: all 0.3s ease;
        text-align: right;
        position: relative;
        overflow: hidden;
    }
    .hero:hover {
        box-shadow: 0 20px 48px rgba(30,58,138,0.32);
    }
    .hero .icon {
        font-size: 40px;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .hero h1 {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        margin: 0.5rem 0 0.2rem;
        line-height: 1.3;
    }
    .hero h1 span {
        color: #FCD34D;
    }
    .hero p {
        font-size: 16px;
        color: #dbeafe;
        margin: 0.5rem 0 1.4rem;
        max-width: 480px;
    }
    .hero .hero-badge {
        background: rgba(16,185,129,0.18);
        color: #d1fae5;
        border: 1px solid rgba(52,211,153,0.4);
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
    }
    .doctor-image {
        border-radius: 24px;
        filter: drop-shadow(0 12px 28px rgba(0,0,0,0.25));
        transition: transform 0.3s ease;
        width: 100%;
        max-width: 280px;
        object-fit: contain;
    }
    .doctor-image:hover {
        transform: scale(1.02);
    }
    .hero-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 30px;
        flex-wrap: wrap-reverse;
        position: relative;
        z-index: 2;
        width: 100%;
    }
    .hero-text {
        flex: 1;
        min-width: 280px;
    }
    .hero-visual {
        position: relative;
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 200px;
    }
    .hero-visual .glow {
        position: absolute;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(245,158,11,0.15), transparent 70%);
    }
    .hero-cta {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: #ffffff !important;
        font-weight: 700;
        font-size: 16px;
        padding: 14px 34px;
        border-radius: 40px;
        text-decoration: none !important;
        box-shadow: 0 8px 24px rgba(245,158,11,0.4);
        transition: all 0.3s ease;
        border: none;
    }
    .hero-cta:hover {
        transform: scale(1.04);
        box-shadow: 0 12px 32px rgba(245,158,11,0.5);
    }
    @media (max-width: 700px) {
        .hero {
            padding: 2rem 1.5rem;
            min-height: auto;
        }
        .hero-content {
            flex-direction: column;
            text-align: center;
        }
        .hero-text {
            min-width: auto;
        }
        .hero h1 {
            font-size: 26px;
        }
        .hero p {
            max-width: 100%;
        }
        .hero-cta {
            justify-content: center;
            width: 100%;
        }
        .doctor-image {
            max-width: 180px;
        }
    }

    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 1.5rem 0;
    }
    @media (max-width: 768px) {
        .features-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 480px) {
        .features-grid { grid-template-columns: 1fr; }
    }
    .feature-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.5);
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        animation: fadeUp 0.8s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.08);
        border-color: rgba(245,158,11,0.2);
    }
    .feature-card .icon { font-size: 32px; }
    .feature-card h4 {
        color: #0f172a;
        font-weight: 700;
        margin: 8px 0 4px;
        font-size: 16px;
    }
    .feature-card p {
        color: #64748b;
        font-size: 13px;
        margin: 0;
    }
    
    .how-section {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(8px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255,255,255,0.5);
        animation: fadeUp 0.8s ease;
    }
    .how-section h3 {
        color: #0f172a;
        font-weight: 700;
        font-size: 22px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .how-steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
    }
    @media (max-width: 768px) {
        .how-steps { grid-template-columns: 1fr; }
    }
    .how-step {
        text-align: center;
        padding: 0 10px;
        animation: fadeUp 0.8s ease;
    }
    .how-step .step-num {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 18px;
        margin: 0 auto 10px;
        transition: transform 0.3s;
        box-shadow: 0 4px 12px rgba(245,158,11,0.3);
    }
    .how-step:hover .step-num {
        transform: scale(1.1) rotate(5deg);
    }
    .how-step h5 {
        color: #0f172a;
        font-weight: 700;
        margin: 0 0 4px;
        font-size: 16px;
    }
    .how-step p {
        color: #64748b;
        font-size: 14px;
        margin: 0;
    }
    
    .trust-section {
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
        margin: 1.5rem 0;
        padding: 1.5rem;
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(4px);
        border-radius: 20px;
        animation: fadeUp 1s ease;
    }
    .trust-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: #0f172a;
        font-size: 15px;
    }
    .trust-item .icon { font-size: 24px; }

    .upload-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 2rem 1.5rem;
        text-align: center;
        border: 2px dashed rgba(245,158,11,0.3);
        transition: all 0.4s ease;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.04);
        animation: fadeUp 0.8s ease;
    }
    .upload-card:hover {
        border-color: #F59E0B;
        background: rgba(255,255,255,0.85);
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(245,158,11,0.08);
    }
    .upload-card .icon {
        font-size: 48px;
        animation: float 3s ease-in-out infinite;
    }
    .upload-card h3 {
        font-weight: 700;
        color: #0f172a;
        margin: 10px 0 4px;
        font-size: 20px;
    }
    .upload-card p {
        color: #64748b;
        font-size: 14px;
        margin: 0;
    }

    .preview-container {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(8px);
        border-radius: 24px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.4);
        box-shadow: 0 8px 32px rgba(0,0,0,0.04);
        margin: 1.5rem 0;
        animation: fadeUp 0.6s ease;
    }
    .preview-container img {
        border-radius: 16px;
        max-height: 350px;
        object-fit: contain;
        width: 100%;
    }

    .scan-container {
        position: relative;
        overflow: hidden;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .scan-container img {
        width: 100%;
        display: block;
        border-radius: 16px;
    }
    .scan-line {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: #dc2626;
        box-shadow: 0 0 20px #dc2626, 0 0 60px #dc2626;
        animation: scanMove 2s ease-in-out infinite;
        z-index: 10;
        border-radius: 2px;
    }
    @keyframes scanMove {
        0% { top: 0; opacity: 1; }
        50% { top: 100%; opacity: 0.8; }
        100% { top: 0; opacity: 1; }
    }
    .scan-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.02);
        pointer-events: none;
        border-radius: 16px;
        border: 2px solid rgba(220, 38, 38, 0.3);
    }

    .result-card {
        border-radius: 24px;
        padding: 1.8rem;
        margin-top: 0.5rem;
        border-left: 6px solid #e11d48;
        animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(4px);
    }
    .result-card.positive {
        background: rgba(254, 242, 242, 0.85);
        border-left-color: #dc2626;
        box-shadow: 0 0 30px rgba(220, 38, 38, 0.08);
    }
    .result-card.negative {
        background: rgba(240, 253, 244, 0.85);
        border-left-color: #16a34a;
        box-shadow: 0 0 30px rgba(22, 163, 74, 0.08);
    }
    .result-card h2 {
        font-size: 30px;
        font-weight: 700;
        margin: 0 0 6px;
    }
    .result-card .confidence {
        font-size: 18px;
        font-weight: 600;
        color: #1e293b;
    }
    .result-card .sub {
        font-size: 14px;
        color: #64748b;
    }
    
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #0f172a;
        margin: 2rem 0 1rem;
        padding-bottom: 8px;
        border-bottom: 3px solid #F59E0B;
        display: inline-block;
        animation: fadeUp 0.6s ease;
    }
    .stImage img { border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
    
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(4px);
        border-radius: 16px;
        padding: 12px 16px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981, #F59E0B, #dc2626) !important;
        height: 12px !important;
        border-radius: 20px !important;
        transition: width 0.8s ease !important;
    }
    .stProgress > div > div {
        background: #e2e8f0 !important;
        border-radius: 20px !important;
        height: 12px !important;
    }
    
    #MainMenu, footer, .stDeployButton { display: none; }
    
    .stButton > button {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 14px 32px;
        font-weight: 700;
        font-size: 16px;
        transition: 0.3s;
        width: 100%;
        box-shadow: 0 4px 16px rgba(245,158,11,0.35);
        animation: fadeUp 0.6s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(245,158,11,0.5);
    }
    
    .disclaimer {
        background: rgba(254, 252, 232, 0.85);
        backdrop-filter: blur(4px);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        border-left: 6px solid #f59e0b;
        margin-top: 2rem;
        font-size: 13px;
        color: #4b5563;
        line-height: 1.6;
        animation: fadeUp 1s ease;
    }
    
    .doctor-btn {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        cursor: not-allowed;
        opacity: 0.7;
        width: 100%;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(245,158,11,0.2);
        margin-top: 10px;
    }
    .doctor-btn:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }
    .coming-badge {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 14px;
        border-radius: 30px;
        margin-left: 8px;
        letter-spacing: 0.5px;
        animation: pulse-badge 1.5s ease-in-out infinite;
    }
    @keyframes pulse-badge {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    /* ===== قائمة التنقل الجانبية ===== */
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    div[data-testid="stRadio"] label {
        background: rgba(255,255,255,0.4);
        padding: 10px 14px !important;
        border-radius: 12px !important;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        font-weight: 500;
        color: #334155;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(245,158,11,0.04);
        border-color: rgba(245,158,11,0.2);
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:last-child {
        margin-left: 0 !important;
        width: 100%;
    }
    div[data-testid="stRadio"] label[data-testid="stRadioLabel"]:has(input:checked) {
        background: rgba(245,158,11,0.12) !important;
        color: #D97706 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(245,158,11,0.2) !important;
    }
    div[data-testid="stRadio"] label input {
        display: none !important;
    }

    /* ===== تنسيق بطاقات المعلومات والنصائح ===== */
    .info-card {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.4);
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        animation: fadeUp 0.6s ease;
    }
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.08);
        border-color: rgba(245,158,11,0.3);
    }
    .info-card h4 {
        color: #0f172a;
        font-weight: 700;
        margin: 0 0 6px 0;
        font-size: 18px;
    }
    .info-card p {
        color: #334155;
        font-size: 15px;
        line-height: 1.6;
        margin: 0;
    }

    .history-card {
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(8px);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        border-left: 6px solid #10b981;
        margin-bottom: 0.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        animation: fadeUp 0.4s ease;
    }
    .history-card:hover {
        transform: scale(1.01);
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }
    .history-card.positive { border-left-color: #dc2626; }
    .history-card .h-left {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .history-card .h-left .h-icon { font-size: 24px; }
    .history-card .h-left .h-date { font-size: 13px; color: #64748b; }
    .history-card .h-left .h-diagnostic {
        font-weight: 700;
        font-size: 16px;
    }
    .history-card .h-right {
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }
    .history-card .h-right .h-conf {
        background: rgba(0,0,0,0.04);
        padding: 4px 12px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 14px;
    }
    .history-card .h-right .h-prob {
        background: rgba(245,158,11,0.1);
        padding: 4px 12px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 14px;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 1rem 0 1.5rem;
    }
    @media (max-width: 640px) {
        .stats-grid { grid-template-columns: 1fr 1fr; }
    }
    .stat-card {
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(4px);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
        transition: 0.2s;
    }
    .stat-card:hover { transform: translateY(-2px); }
    .stat-card .stat-value {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
    }
    .stat-card .stat-label {
        font-size: 13px;
        color: #64748b;
        margin-top: 4px;
    }
    .stat-card .stat-icon { font-size: 28px; margin-bottom: 4px; }

    @media (max-width: 640px) {
        .header-left h1 { font-size: 18px; }
        .header-left .subtitle { font-size: 11px; }
        .header-ai-badge { font-size: 10px; padding: 3px 10px; }
        .top-badge-card { padding: 10px; gap: 8px; }
        .top-badge-card .tb-icon { width: 34px; height: 34px; min-width: 34px; font-size: 15px; }
        .top-badge-card .tb-title { font-size: 12px; }
        .top-badge-card .tb-desc { font-size: 10px; }
        .hero { padding: 1.5rem 1rem; min-height: auto; }
        .hero h1 { font-size: 22px; }
        .hero p { font-size: 14px; }
        .feature-card { padding: 1rem; }
        .feature-card .icon { font-size: 28px; }
        .how-section { padding: 1.5rem; }
        .sidebar-glass { padding: 1rem; }
        .result-card { padding: 1.2rem; }
        .result-card h2 { font-size: 24px; }
        .stButton > button { padding: 12px 20px; font-size: 14px; }
        div[data-testid="stRadio"] label { padding: 8px 12px !important; font-size: 14px; }
    }
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE ==========
if 'language' not in st.session_state:
    st.session_state.language = "fr"
if 'history' not in st.session_state:
    st.session_state.history = []
if 'upload_mode' not in st.session_state:
    st.session_state.upload_mode = "file"
if 'page' not in st.session_state:
    st.session_state.page = "home"

def get_file_base64(names):
    for name in names:
        if os.path.exists(name):
            with open(name, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

logo = get_file_base64(["logo.png", "logo.jpg", "logo.jpeg", "LOGO.png"])
logo_icon = get_file_base64(["logo_icon.png", "logo-icon.png"]) or logo

# ========== SIDEBAR ==========
with st.sidebar:
    if logo:
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:1rem;">
            <img src="data:image/png;base64,{logo}" style="max-width:170px; width:100%;">
        </div>
        """, unsafe_allow_html=True)

    lang_map = {
        "ar": "🇸🇦 العربية",
        "fr": "🇫🇷 Français",
        "en": "🇬🇧 English"
    }
    selected_lang = st.selectbox(
        t("lang_selector"),
        options=list(lang_map.keys()),
        format_func=lambda x: lang_map[x],
        index=list(lang_map.keys()).index(st.session_state.language)
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()
    
    st.markdown("---")
    
    # ===== قائمة التنقل الموسعة =====
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>📋 القائمة / Menu</h4>
    </div>
    """, unsafe_allow_html=True)
    
    nav_options = [
        t("nav_home"),
        t("nav_blood_analysis"),
        t("nav_history"),
        t("nav_disease_info"),
        t("nav_tips"),
        t("nav_about")
    ]
    # تعيين الفهرس الحالي
    current_idx = 0
    if st.session_state.page == "blood_analysis":
        current_idx = 1
    elif st.session_state.page == "history":
        current_idx = 2
    elif st.session_state.page == "disease_info":
        current_idx = 3
    elif st.session_state.page == "tips":
        current_idx = 4
    elif st.session_state.page == "about":
        current_idx = 5
    
    nav_choice = st.radio(
        "##",
        options=nav_options,
        index=current_idx,
        label_visibility="collapsed",
        key="nav_radio"
    )
    
    if nav_choice == t("nav_home"):
        st.session_state.page = "home"
    elif nav_choice == t("nav_blood_analysis"):
        st.session_state.page = "blood_analysis"
    elif nav_choice == t("nav_history"):
        st.session_state.page = "history"
    elif nav_choice == t("nav_disease_info"):
        st.session_state.page = "disease_info"
    elif nav_choice == t("nav_tips"):
        st.session_state.page = "tips"
    elif nav_choice == t("nav_about"):
        st.session_state.page = "about"
    
    # ===== باقي عناصر القائمة =====
    APP_URL = "https://hwaxrexkahkxaazwwjjr3d.streamlit.app/"
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&margin=10&color=F59E0B&data={APP_URL}"
    
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>{t('sidebar_qr_title')}</h4>
        <div class="qr-container">
            <img src="{qr_api_url}" width="160" height="160" alt="QR Code">
        </div>
        <p style="text-align:center; font-size:13px; color:#64748b;">{t('sidebar_qr_desc')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>{t('sidebar_health_title')}</h4>
        <ul>
            <li>{t('sidebar_health_1')}</li>
            <li>{t('sidebar_health_2')}</li>
            <li>{t('sidebar_health_3')}</li>
            <li>{t('sidebar_health_4')}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>{t('sidebar_doctor_title')} <span class="coming-badge">{t('sidebar_doctor_soon')}</span></h4>
        <p>{t('sidebar_doctor_desc')}</p>
        <div class="doctor-btn">{t('sidebar_doctor_btn')}</div>
        <p style="font-size:12px; color:#94a3b8; text-align:center; margin-top:8px;">{t('sidebar_doctor_soon')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption(t('sidebar_version'))

# ========== HEADER العلوي ==========
logo_html = f'<div class="header-logo-badge"><img src="data:image/png;base64,{logo_icon}"></div>' if logo_icon else '<div class="header-logo-badge" style="font-size:26px;">🧬</div>'

st.markdown(f"""
<div class="header">
    <div class="header-badges">
        <div class="header-icon-btn">🔔<span class="dot"></span></div>
        <div class="header-badge" style="background:rgba(255,255,255,0.15); color:#fff; border-color:rgba(255,255,255,0.25);">🌐 {lang_map[st.session_state.language]}</div>
        <div class="header-ai-badge">✅ {t('ai_dev_badge')}</div>
    </div>
    <div class="header-left">
        <div>
            <h1>{t('app_title')}</h1>
            <div class="subtitle">{t('app_subtitle')}</div>
        </div>
        {logo_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ========== TOP BADGES ==========
st.markdown(f"""
<div class="top-badges">
    <div class="top-badge-card">
        <div class="tb-icon">🔒</div>
        <div><div class="tb-title">{t('badge_private')}</div><div class="tb-desc">{t('badge_private_desc')}</div></div>
    </div>
    <div class="top-badge-card">
        <div class="tb-icon">🕐</div>
        <div><div class="tb-title">{t('badge_available')}</div><div class="tb-desc">{t('badge_available_desc')}</div></div>
    </div>
    <div class="top-badge-card">
        <div class="tb-icon">📞</div>
        <div><div class="tb-title">{t('badge_call')}</div><div class="tb-desc">{t('badge_call_desc')}</div></div>
    </div>
    <div class="top-badge-card">
        <div class="tb-icon">🎧</div>
        <div><div class="tb-title">{t('badge_support')}</div><div class="tb-desc">{t('badge_support_desc')}</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== دوال مساعدة للتحليل ==========
def clean_mask(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(cleaned, [c], -1, 255, -1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

def enhance_conjunctiva(image):
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l_enh = clahe.apply(l)
        l_enh = cv2.GaussianBlur(l_enh, (3,3), 0)
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        l_enh = cv2.filter2D(l_enh, -1, kernel)
        lab_enh = cv2.merge((l_enh, a, b))
        enhanced = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2RGB)
        return enhanced
    return image

def extract_best_conjunctiva(img, mask):
    mask = clean_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 15
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img.shape[1] - x, w + 2*pad)
        h = min(img.shape[0] - y, h + 2*pad)
        if w > 0 and h > 0:
            cropped = img[y:y+h, x:x+w]
            enhanced = enhance_conjunctiva(cropped)
            mask_cropped = mask[y:y+h, x:x+w]
            return enhanced, mask_cropped, (x, y, w, h)
    conj = cv2.bitwise_and(img, img, mask=mask)
    enhanced = enhance_conjunctiva(conj)
    return enhanced, mask, None

def predict_anemia(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        raw_pred = torch.sigmoid(output).item()
    
    corrected_pred = 1 - raw_pred
    
    if corrected_pred >= 0.5:
        result = "Anemic"
        confidence = corrected_pred * 100
    else:
        result = "Non Anemic"
        confidence = (1 - corrected_pred) * 100
    
    return result, confidence, corrected_pred, raw_pred

@st.cache_resource
def load_models():
    unet, dev_unet = load_unet_model()
    clf, dev_clf = load_classifier_model()
    return unet, dev_unet, clf, dev_clf

# ================================================================
# ========== الصفحات حسب الاختيار ==========
# ================================================================

if st.session_state.page == "home" or st.session_state.page == "blood_analysis":
    # ===== الصفحة الرئيسية / تحليل صورة الدم =====
    # نعرض الهيرو فقط في الصفحة الرئيسية
    if st.session_state.page == "home":
        doctor_local = get_file_base64(["doctor.png", "doctor.jpg", "doctor.jpeg"])
        if doctor_local:
            doctor_img_url = f"data:image/png;base64,{doctor_local}"
        else:
            doctor_img_url = "https://cdn-icons-png.flaticon.com/512/3774/3774299.png"
        heartbeat_svg = (
            "data:image/svg+xml;utf8,"
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 60'>"
            "<polyline points='0,30 60,30 80,10 100,50 120,5 140,55 160,30 220,30 240,15 260,45 280,30 400,30' "
            "fill='none' stroke='rgba(255,255,255,0.3)' stroke-width='3'/></svg>"
        )
        st.markdown(f"""
        <div class="hero" id="hero-top">
            <img src="{heartbeat_svg}" class="hero-heartbeat" alt="">
            <div class="hero-content">
                <div class="hero-text">
                    <span class="icon">🩺</span>
                    <h1>{t('hero_title')}</h1>
                    <p>{t('hero_desc')}</p>
                    <a href="#upload-zone" class="hero-cta">➜ {t('hero_cta')}</a>
                    <div style="margin-top:16px;">
                        <span class="hero-badge">✅ {t('hero_badge')}</span>
                    </div>
                </div>
                <div class="hero-visual">
                    <div class="glow"></div>
                    <img src="{doctor_img_url}" class="doctor-image" alt="Doctor">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # في صفحة تحليل صورة الدم نعرض عنوان بسيط
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a8a, #2563eb); border-radius: 20px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center; color: white;">
            <h2 style="font-weight: 700; margin:0;">🩸 {t('nav_blood_analysis')}</h2>
            <p style="margin:0; opacity:0.8;">{t('upload_desc')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== واجهة الرفع =====
    st.markdown(f"""
    <div class="upload-card" id="upload-zone">
        <div class="icon">🔬</div>
        <h3>{t('upload_title')}</h3>
        <p>{t('upload_desc')}</p>
    </div>
    """, unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📁 " + t("upload_method_file"), use_container_width=True,
                     type="primary" if st.session_state.upload_mode == "file" else "secondary"):
            st.session_state.upload_mode = "file"
            st.rerun()
    with col_btn2:
        if st.button("📷 " + t("upload_method_camera"), use_container_width=True,
                     type="primary" if st.session_state.upload_mode == "camera" else "secondary"):
            st.session_state.upload_mode = "camera"
            st.rerun()

    if st.session_state.upload_mode == "file":
        uploaded = st.file_uploader(
            t("upload_file_label"),
            type=["jpg","png","jpeg"],
            label_visibility="collapsed",
            key="file_uploader"
        )
    else:
        uploaded = st.camera_input(
            t("upload_camera_label"),
            label_visibility="collapsed",
            key="camera_input"
        )

    # ===== معالجة الصورة =====
    if uploaded is not None:
        preview_placeholder = st.empty()
        with preview_placeholder.container():
            st.markdown(f"""
            <div class="preview-container">
                <h4 style="margin-top:0;">{t('preview_title')}</h4>
            """, unsafe_allow_html=True)
            col_preview, _ = st.columns([1, 1])
            with col_preview:
                st.image(uploaded, use_container_width=True, caption=t('preview_caption'))
            st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button(t("analyze_btn"), use_container_width=True):
            img_pil = Image.open(uploaded).convert('RGB')
            img_b64 = image_to_base64(img_pil)
            
            scan_placeholder = st.empty()
            with scan_placeholder.container():
                st.markdown(f"""
                <div class="preview-container">
                    <h4 style="margin-top:0;">🔬 جاري المسح الضوئي...</h4>
                    <div class="scan-container">
                        <img src="data:image/jpeg;base64,{img_b64}" style="width:100%; border-radius:16px;">
                        <div class="scan-line"></div>
                        <div class="scan-overlay"></div>
                    </div>
                    <p style="text-align:center; color:#64748b; margin-top:8px;">{t('analyzing')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner(t("loading_models")):
                unet_model, unet_device, clf_model, clf_device = load_models()
            
            progress_bar = st.progress(0)
            for i in range(10):
                time.sleep(0.06)
                progress_bar.progress((i+1)*10)
            
            with st.spinner(t("analyzing")):
                img = np.array(img_pil)
                
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

                result, confidence, corrected_pred, raw_pred = predict_anemia(clf_model, conj_enhanced, clf_device)

                anemia_pct = corrected_pred * 100
                non_pct = (1 - corrected_pred) * 100

                progress_bar.empty()
                scan_placeholder.empty()
                st.success(t("analysis_done"))

                # ===== عرض النتائج =====
                st.markdown(f'<div class="section-title">{t("results_title")}</div>', unsafe_allow_html=True)
                
                col_left, col_right = st.columns([2, 1])
                with col_left:
                    st.markdown(f"**👁️ {t('result_conjunctiva')}**")
                    st.image(conj_enhanced, use_container_width=True)
                with col_right:
                    st.markdown(f"**{t('result_original')}**")
                    st.image(img, use_container_width=True)
                    st.markdown(f"**{t('result_mask')}**")
                    st.image(final_mask, use_container_width=True, clamp=True)

                before = np.sum(raw_mask > 0) / 255
                after = np.sum(final_mask > 0) / 255
                reduction = ((before - after) / before * 100) if before > 0 else 0

                m1, m2 = st.columns(2)
                with m1:
                    st.metric(t("metric_surface"), f"{after:.0f} px²")
                with m2:
                    st.metric(t("metric_cleaning"), f"{reduction:.1f}%")

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

                # ===== إضافة إلى السجل =====
                entry = {
                    t("history_date"): datetime.now().strftime("%Y-%m-%d %H:%M"),
                    t("history_diagnostic"): result,
                    t("history_confidence"): f"{confidence:.1f}%",
                    t("history_prob"): f"{anemia_pct:.1f}%",
                    "prob_value": anemia_pct
                }
                st.session_state.history.append(entry)
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop(0)

                # ===== التفاصيل التقنية =====
                with st.expander(t("tech_details")):
                    st.write(f"**{t('tech_model_seg')}:** U‑Net (ResNet34)")
                    st.write(f"**{t('tech_model_clf')}:** EfficientNet‑B3")
                    st.write(f"**{t('tech_device')}:** {'GPU' if clf_device.type == 'cuda' else 'CPU'}")
                    st.write(f"**{t('tech_sigmoid')}:** {raw_pred:.4f}")
                    st.write(f"**{t('tech_prob_anemic')}:** {anemia_pct:.1f}%")
                    st.write(f"**{t('tech_prob_non')}:** {non_pct:.1f}%")
                    st.write(f"**{t('tech_preprocess')}:** CLAHE + Filtrage + Netteté")
                    st.write(f"**{t('tech_decision')}:** {t('tech_decision_value')}")

                st.markdown(f"""
                <div class="disclaimer">
                    <strong>{t('disclaimer')}</strong><br>
                    {t('disclaimer_text')}<br>
                    {t('disclaimer_consult')}
                </div>
                """, unsafe_allow_html=True)

    # ===== الميزات =====
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <div class="icon">⚡</div>
            <h4>""" + t('feature_1_title') + """</h4>
            <p>""" + t('feature_1_desc') + """</p>
        </div>
        <div class="feature-card">
            <div class="icon">🩸</div>
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

    # ===== كيف يعمل التطبيق =====
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

    # ===== شارات الثقة =====
    st.markdown(f"""
    <div class="trust-section">
        <div class="trust-item"><span class="icon">❤️</span> {t('trust_1')}</div>
        <div class="trust-item"><span class="icon">⭐</span> {t('trust_2')}</div>
        <div class="trust-item"><span class="icon">⚡</span> {t('trust_3')}</div>
        <div class="trust-item"><span class="icon">🧠</span> {t('trust_4')}</div>
        <div class="trust-item"><span class="icon">📱</span> {t('version')}</div>
        <div class="trust-item"><span class="icon">🎯</span> {t('accuracy')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ===== معلومات إضافية (سرعة، ذكاء، أمان) =====
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

# ================================================================
# ========== صفحة النتائج السابقة ==========
# ================================================================
elif st.session_state.page == "history":
    st.markdown(f'<div class="section-title">{t("history_title")}</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        total = len(st.session_state.history)
        positive = sum(1 for h in st.session_state.history if h[t("history_diagnostic")] == "Anemic")
        negative = total - positive
        avg_conf = sum(float(h[t("history_confidence")].replace('%','')) for h in st.session_state.history) / total if total > 0 else 0
        
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">📊</div>
                <div class="stat-value">{total}</div>
                <div class="stat-label">{t('history_total')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🩸</div>
                <div class="stat-value" style="color:#dc2626;">{positive}</div>
                <div class="stat-label">{t('history_positive')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">✅</div>
                <div class="stat-value" style="color:#16a34a;">{negative}</div>
                <div class="stat-label">{t('history_negative')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📈</div>
                <div class="stat-value">{avg_conf:.1f}%</div>
                <div class="stat-label">{t('history_avg_conf')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if total >= 2:
            st.markdown(f'<h4 style="margin-top:1rem;">{t("history_chart_title")}</h4>', unsafe_allow_html=True)
            dates = []
            probs = []
            for h in st.session_state.history:
                date_str = h.get(t("history_date"), "")
                prob_str = h.get(t("history_prob"), "0%").replace('%','')
                try:
                    prob = float(prob_str)
                except:
                    prob = 0
                dates.append(date_str)
                probs.append(prob)
            
            if len(dates) >= 2:
                fig2, ax2 = plt.subplots(figsize=(8,4))
                ax2.plot(dates, probs, marker='o', linestyle='-', color='#dc2626', linewidth=2, markersize=6)
                ax2.set_ylim(0,100)
                ax2.set_ylabel("Probabilité (%)")
                ax2.set_xlabel("Date")
                ax2.grid(True, alpha=0.3)
                ax2.set_title(t("history_chart_title"), fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig2)
        
        st.markdown("---")
        st.markdown("#### 📋 تفاصيل التحاليل")
        
        for h in reversed(st.session_state.history):
            diagnostic = h[t("history_diagnostic")]
            is_positive = diagnostic == "Anemic"
            date = h[t("history_date")]
            conf = h[t("history_confidence")]
            prob = h[t("history_prob")]
            
            st.markdown(f"""
            <div class="history-card {'positive' if is_positive else ''}">
                <div class="h-left">
                    <span class="h-icon">{'🩸' if is_positive else '✅'}</span>
                    <span class="h-date">{date}</span>
                    <span class="h-diagnostic" style="color:{'#dc2626' if is_positive else '#16a34a'};">
                        {diagnostic}
                    </span>
                </div>
                <div class="h-right">
                    <span class="h-conf">🔹 {conf}</span>
                    <span class="h-prob">📊 {prob}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        col_export, col_clear = st.columns(2)
        with col_export:
            if st.button(t("history_export"), use_container_width=True):
                st.info("📥 سيتم تصدير التقرير قريباً... (وهمي)")
        with col_clear:
            if st.button(t("history_clear"), use_container_width=True):
                st.session_state.history = []
                st.rerun()
    else:
        st.markdown(f"""
        <div style="text-align:center; padding: 4rem 1rem; background: rgba(255,255,255,0.6); border-radius: 32px; backdrop-filter: blur(4px); border: 2px dashed rgba(245,158,11,0.2);">
            <div style="font-size: 64px; margin-bottom: 1rem;">📭</div>
            <h3 style="color: #0f172a; font-weight: 700;">{t('history_empty')}</h3>
            <p style="color: #64748b; font-size: 16px;">{t('history_empty_desc')}</p>
            <br>
            <a href="#upload-zone" class="hero-cta" style="display: inline-flex;">➜ {t('hero_cta')}</a>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# ========== صفحة معلومات المرض ==========
# ================================================================
elif st.session_state.page == "disease_info":
    st.markdown(f'<div class="section-title">{t("disease_info_title")}</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-card">
        <h4>📌 نظرة عامة</h4>
        <p>{t('disease_info_desc')}</p>
    </div>
    <div class="info-card">
        <h4>⚠️ الأعراض</h4>
        <p>{t('disease_info_symptoms')}</p>
    </div>
    <div class="info-card">
        <h4>🔍 الأسباب</h4>
        <p>{t('disease_info_causes')}</p>
    </div>
    <div class="info-card">
        <h4>🛡️ الوقاية</h4>
        <p>{t('disease_info_prevention')}</p>
    </div>
    <div class="info-card">
        <h4>💊 العلاج</h4>
        <p>{t('disease_info_treatment')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(245,158,11,0.08); border-radius: 16px; padding: 1rem; border-left: 4px solid #F59E0B; margin-top: 1rem;">
        <p style="margin:0; color:#0f172a; font-weight:500;">💡 تذكر أن هذه المعلومات هي لأغراض توعوية ولا تغني عن استشارة الطبيب المختص.</p>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# ========== صفحة نصائح صحية ==========
# ================================================================
elif st.session_state.page == "tips":
    st.markdown(f"""
    <div style="text-align:center; margin-bottom: 1.5rem;">
        <h2 style="font-weight: 800; color: #0f172a; font-size: 28px;">{t('nav_tips')}</h2>
        <p style="color: #64748b; font-size: 16px;">{t('tips_subtitle') if 'tips_subtitle' in LANGUAGES[st.session_state.language] else ''}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # قائمة النصائح من الترجمة
    tips_list = t("tips_list") if "tips_list" in LANGUAGES[st.session_state.language] else []
    
    if tips_list:
        # نصيحة عشوائية
        if 'random_tip_index' not in st.session_state:
            st.session_state.random_tip_index = random.randint(0, len(tips_list)-1)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a8a, #2563eb); border-radius: 20px; padding: 1.5rem; margin: 1rem 0; color: white; text-align: center; box-shadow: 0 8px 24px rgba(30,58,138,0.25); animation: fadeUp 0.8s ease;">
            <div style="font-size: 48px;">{tips_list[st.session_state.random_tip_index]['icon']}</div>
            <div style="font-size: 18px; font-weight: 500; margin: 10px 0;">“{tips_list[st.session_state.random_tip_index]['text']}”</div>
            <div style="font-size: 14px; opacity: 0.8; background: rgba(255,255,255,0.15); padding: 4px 16px; border-radius: 30px; display: inline-block;">#{tips_list[st.session_state.random_tip_index]['category']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(t("tips_random_btn") if "tips_random_btn" in LANGUAGES[st.session_state.language] else "🎲 نصيحة عشوائية", use_container_width=True):
            st.session_state.random_tip_index = random.randint(0, len(tips_list)-1)
            st.rerun()
    
    st.markdown("#### 🔍 تصفية النصائح")
    categories = ["all", "nutrition", "vitamins", "lifestyle", "followup"]
    cat_labels = {
        "all": "📌 الكل",
        "nutrition": t("tips_category_nutrition") if "tips_category_nutrition" in LANGUAGES[st.session_state.language] else "🍎 التغذية",
        "vitamins": t("tips_category_vitamins") if "tips_category_vitamins" in LANGUAGES[st.session_state.language] else "💊 الفيتامينات",
        "lifestyle": t("tips_category_lifestyle") if "tips_category_lifestyle" in LANGUAGES[st.session_state.language] else "🧘 نمط الحياة",
        "followup": t("tips_category_followup") if "tips_category_followup" in LANGUAGES[st.session_state.language] else "🩺 المتابعة الطبية"
    }
    
    if 'tip_filter' not in st.session_state:
        st.session_state.tip_filter = "all"
    
    col_filters = st.columns(len(categories))
    for i, cat in enumerate(categories):
        with col_filters[i]:
            active = st.session_state.tip_filter == cat
            if st.button(cat_labels[cat], use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.tip_filter = cat
                st.rerun()
    
    search_query = st.text_input("", placeholder=t("tips_search_placeholder") if "tips_search_placeholder" in LANGUAGES[st.session_state.language] else "🔍 ابحث في النصائح...", label_visibility="collapsed")
    
    filtered_tips = tips_list
    if st.session_state.tip_filter != "all":
        filtered_tips = [tip for tip in tips_list if tip['category'] == st.session_state.tip_filter]
    if search_query:
        filtered_tips = [tip for tip in filtered_tips if search_query.lower() in tip['text'].lower()]
    
    if filtered_tips:
        for tip in filtered_tips:
            cat_class = tip['category']
            cat_label = cat_labels.get(cat_class, cat_class)
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.75); backdrop-filter: blur(8px); border-radius: 20px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.4); transition: all 0.3s ease; box-shadow: 0 4px 16px rgba(0,0,0,0.04); display: flex; align-items: center; gap: 16px; animation: fadeUp 0.6s ease;">
                <div style="font-size: 32px; min-width: 48px; text-align: center;">{tip['icon']}</div>
                <div style="flex: 1; color: #0f172a; font-size: 15px; line-height: 1.5;">{tip['text']}</div>
                <span style="font-size: 12px; font-weight: 600; background: rgba(245,158,11,0.1); color: #D97706; padding: 2px 12px; border-radius: 30px; white-space: nowrap;">{cat_label}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("🚫 لا توجد نصائح تطابق معايير البحث.")
    
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(245,158,11,0.08); border-radius: 16px; padding: 1rem; border-left: 4px solid #F59E0B; margin-top: 1rem;">
        <p style="margin:0; color:#0f172a; font-weight:500;">💡 تذكر أن هذه النصائح هي لأغراض توعوية ولا تغني عن استشارة الطبيب المختص.</p>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# ========== صفحة حول التطبيق ==========
# ================================================================
elif st.session_state.page == "about":
    st.markdown(f'<div class="section-title">{t("about_title")}</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-card">
        <h4>📌 {t('about_title')}</h4>
        <p>{t('about_desc')}</p>
    </div>
    <div class="info-card">
        <h4>🎯 مهمتنا</h4>
        <p>{t('about_mission')}</p>
    </div>
    <div class="info-card">
        <h4>👨‍💻 فريق التطوير</h4>
        <p>{t('about_team')}</p>
    </div>
    <div class="info-card">
        <h4>📧 التواصل</h4>
        <p>{t('about_contact')}</p>
    </div>
    <div class="info-card">
        <h4>📱 الإصدار</h4>
        <p>{t('about_version')}</p>
    </div>
    <div class="info-card">
        <h4>⚙️ التقنيات المستخدمة</h4>
        <p>{t('about_tech')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(16,185,129,0.08); border-radius: 16px; padding: 1rem; border-left: 4px solid #10b981; margin-top: 1rem;">
        <p style="margin:0; color:#0f172a; font-weight:500;">❤️ شكراً لاستخدامك AnemiCheck – صحتك تهمنا.</p>
    </div>
    """, unsafe_allow_html=True)

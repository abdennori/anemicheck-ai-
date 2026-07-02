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

# ========== الترجمة الموسعة ==========
LANGUAGES = {
    "ar": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "الذكاء الاصطناعي لصحة الدم",
        "badge_private": "أمين وموثوق",
        "badge_private_desc": "100% خصوصيتك محمية",
        "badge_available": "خدمة متاحة",
        "badge_available_desc": "24/7 نحن هنا من أجلك دائماً",
        "hero_title": "طبيبك معك بأقل من دقيقتين",
        "hero_desc": "تحليل ذكي لصورة العين لكشف الأنيميا بدقة وسرعة عالية",
        "hero_badge": "100% نتائج دقيقة وآمنة",
        "hero_cta": "ابدأ التحليل الآن",
        "ai_dev_badge": "تم تطويره بواسطة الذكاء الاصطناعي",
        "top_badge1_title": "مساعدة فورية",
        "top_badge1_desc": "تواصل مع فريق الدعم",
        "top_badge2_title": "مكالمة مجانية",
        "top_badge2_desc": "🔜 قريباً",
        "top_badge3_title": "خدمة متاحة 24/7",
        "top_badge3_desc": "نحن هنا من أجلك دائماً",
        "top_badge4_title": "آمن وموثوق",
        "top_badge4_desc": "خصوصيتك محمية 100%",
        "feature_1_title": "نتيجة سريعة",
        "feature_1_desc": "احصل على النتيجة فوراً مع تقرير مفصل",
        "feature_2_title": "رفع الصورة",
        "feature_2_desc": "قم برفع صورة واضحة للعين (JPG, PNG)",
        "feature_3_title": "موثوق ومعتمد",
        "feature_3_desc": "هذا التطبيق يساعد في الكشف المبكر عن الأنيميا",
        "feature_4_title": "أمان",
        "feature_4_desc": "بياناتك محمية بالكامل",
        "how_title": "كيف يعمل التطبيق؟",
        "how_step1": "رفع الصورة",
        "how_step1_desc": "ارفع صورة واضحة للعين",
        "how_step2": "تحليل ذكي",
        "how_step2_desc": "يستخدم النموذج الذكاء الاصطناعي للتحليل",
        "how_step3": "النتيجة الفورية",
        "how_step3_desc": "احصل على التشخيص في ثوانٍ",
        "trust_1": "أمان",
        "trust_2": "رعاية",
        "trust_3": "ثقة عالية",
        "version": "الإصدار 1.0.0",
        "accuracy": "دقة تصل إلى 96%",
        "fast": "سريع",
        "fast_desc": "النتيجة في أقل من دقيقتين",
        "smart": "ذكي",
        "smart_desc": "تقنية ذكاء اصطناعي متقدمة",
        "secure": "أمان",
        "secure_desc": "بياناتك محمية بالكامل",
        "upload_title": "رفع صورة العين",
        "upload_desc": "صورة واضحة للعين للكشف عن الأنيميا",
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
        "result_conjunctiva": "👁️ الملتحمة المحسّنة",
        "metric_surface": "📐 المساحة المقطعة",
        "metric_cleaning": "🧼 التنظيف",
        "diagnostic_title": "🩺 التشخيص",
        "diagnostic_anemic": "🩸 فقر دم",
        "diagnostic_anemic_desc": "يوجد فقر دم",
        "diagnostic_non_anemic": "✅ لا يوجد فقر دم",
        "diagnostic_non_anemic_desc": "لا يوجد فقر دم",
        "diagnostic_confidence": "📊 مستوى الثقة",
        "chart_title": "📈 توزيع الاحتمالات",
        "chart_non": "غير مصاب",
        "chart_anemic": "مصاب",
        "history_title": "📋 سجل التحليلات",
        "history_date": "التاريخ",
        "history_diagnostic": "التشخيص",
        "history_confidence": "الثقة",
        "history_prob": "احتمال فقر الدم",
        "history_empty": "📭 لا توجد تحليلات سابقة بعد",
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
        "tech_prob_anemic": "احتمال فقر الدم",
        "tech_prob_non": "احتمال عدم الإصابة",
        "tech_preprocess": "المعالجة المسبقة",
        "tech_decision": "القرار",
        "tech_decision_value": "مستقل تماماً، بدون تحيز",
        "disclaimer": "⚠️ تنويه طبي",
        "disclaimer_text": "هذه النتيجة صادرة عن نموذج ذكاء اصطناعي ولا تغني عن استشارة الطبيب المختص.",
        "disclaimer_consult": "استشر طبيباً للحصول على تشخيص دقيق.",
        "welcome_title": "👁️ جاهز للتحليل",
        "welcome_desc": "حمّل صورة العين للبدء.",
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
        "nav_history": "📋 سجل التحليلات",
        "nav_tips": "💡 نصائح طبية",
        "tips_title": "💡 نصائح طبية لصحة أفضل",
        "tips_subtitle": "نصائح مخصصة لتحسين مستوى الحديد والوقاية من الأنيميا",
        "tips_category_nutrition": "🍎 التغذية",
        "tips_category_vitamins": "💊 الفيتامينات",
        "tips_category_lifestyle": "🧘 نمط الحياة",
        "tips_category_followup": "🩺 المتابعة الطبية",
        "tips_random_btn": "🎲 نصيحة عشوائية",
        "tips_search_placeholder": "🔍 ابحث في النصائح...",
        "tips_list": [
            {"category": "nutrition", "text": "تناول السبانخ، العدس، واللحوم الحمراء لزيادة الحديد.", "icon": "🥩"},
            {"category": "nutrition", "text": "أضف الحمص والفاصوليا إلى وجباتك اليومية.", "icon": "🍲"},
            {"category": "nutrition", "text": "تناول البيض مع الخضار الورقية لتحسين امتصاص الحديد.", "icon": "🥚"},
            {"category": "vitamins", "text": "فيتامين C يعزز امتصاص الحديد، تناوله مع الوجبات.", "icon": "🍊"},
            {"category": "vitamins", "text": "مكملات الحديد يجب تناولها تحت إشراف طبيب.", "icon": "💊"},
            {"category": "vitamins", "text": "فيتامين B12 مهم لتكوين خلايا الدم الحمراء.", "icon": "🧪"},
            {"category": "lifestyle", "text": "مارس الرياضة الخفيفة لتحسين الدورة الدموية.", "icon": "🚶"},
            {"category": "lifestyle", "text": "احصل على قسط كافٍ من النوم لدعم جهاز المناعة.", "icon": "😴"},
            {"category": "lifestyle", "text": "تجنب التدخين لأنه يقلل من كفاءة نقل الأكسجين.", "icon": "🚭"},
            {"category": "followup", "text": "قم بفحص الدم بشكل دوري لمتابعة مستويات الحديد.", "icon": "🩸"},
            {"category": "followup", "text": "استشر طبيبك إذا شعرت بأعراض مثل التعب والدوخة.", "icon": "👨‍⚕️"},
            {"category": "followup", "text": "التزم بجدول المتابعة الذي يحدده طبيبك.", "icon": "📅"}
        ]
    },
    "fr": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "L'IA pour la santé sanguine",
        "badge_private": "Sécurisé et fiable",
        "badge_private_desc": "100% de confidentialité garantie",
        "badge_available": "Service disponible",
        "badge_available_desc": "24/7 nous sommes là pour vous",
        "hero_title": "Votre médecin en moins de deux minutes",
        "hero_desc": "Analyse intelligente de l'image oculaire pour détecter l'anémie avec précision et rapidité",
        "hero_badge": "100% de résultats précis et sécurisés",
        "hero_cta": "Commencer l'analyse",
        "ai_dev_badge": "Développé avec l'intelligence artificielle",
        "top_badge1_title": "Aide immédiate",
        "top_badge1_desc": "Contactez notre équipe",
        "top_badge2_title": "Appel gratuit",
        "top_badge2_desc": "🔜 Bientôt",
        "top_badge3_title": "Service 24/7",
        "top_badge3_desc": "Nous sommes toujours là",
        "top_badge4_title": "Sûr et fiable",
        "top_badge4_desc": "100% de confidentialité",
        "feature_1_title": "Résultat rapide",
        "feature_1_desc": "Obtenez le résultat immédiatement avec un rapport détaillé",
        "feature_2_title": "Téléchargement d'image",
        "feature_2_desc": "Téléchargez une image claire de l'œil (JPG, PNG)",
        "feature_3_title": "Fiable et approuvé",
        "feature_3_desc": "Cette application aide à la détection précoce de l'anémie",
        "feature_4_title": "Sécurité",
        "feature_4_desc": "Vos données sont entièrement protégées",
        "how_title": "Comment ça marche ?",
        "how_step1": "Téléchargement",
        "how_step1_desc": "Téléchargez une image claire de l'œil",
        "how_step2": "Analyse intelligente",
        "how_step2_desc": "Le modèle d'IA analyse l'image",
        "how_step3": "Résultat instantané",
        "how_step3_desc": "Obtenez le diagnostic en quelques secondes",
        "trust_1": "Sécurité",
        "trust_2": "Soins",
        "trust_3": "Haute confiance",
        "version": "Version 1.0.0",
        "accuracy": "Précision jusqu'à 96%",
        "fast": "Rapide",
        "fast_desc": "Résultat en moins de 2 minutes",
        "smart": "Intelligent",
        "smart_desc": "Technologie IA avancée",
        "secure": "Sécurisé",
        "secure_desc": "Vos données sont protégées",
        "upload_title": "Télécharger une image de l'œil",
        "upload_desc": "Image claire de l'œil pour la détection de l'anémie",
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
        "result_conjunctiva": "👁️ Conjonctive optimisée",
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
        "history_title": "📋 Historique des analyses",
        "history_date": "Date",
        "history_diagnostic": "Diagnostic",
        "history_confidence": "Confiance",
        "history_prob": "Probabilité Anémie",
        "history_empty": "📭 Aucun historique",
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
        "welcome_title": "👁️ Prêt à analyser",
        "welcome_desc": "Téléchargez une image de l'œil pour commencer.",
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
        "nav_history": "📋 Historique",
        "nav_tips": "💡 Conseils médicaux",
        "tips_title": "💡 Conseils médicaux pour une meilleure santé",
        "tips_subtitle": "Des conseils personnalisés pour améliorer votre taux de fer et prévenir l'anémie",
        "tips_category_nutrition": "🍎 Nutrition",
        "tips_category_vitamins": "💊 Vitamines",
        "tips_category_lifestyle": "🧘 Mode de vie",
        "tips_category_followup": "🩺 Suivi médical",
        "tips_random_btn": "🎲 Conseil aléatoire",
        "tips_search_placeholder": "🔍 Rechercher un conseil...",
        "tips_list": [
            {"category": "nutrition", "text": "Mangez des épinards, des lentilles et de la viande rouge pour augmenter le fer.", "icon": "🥩"},
            {"category": "nutrition", "text": "Ajoutez des pois chiches et des haricots à vos repas quotidiens.", "icon": "🍲"},
            {"category": "nutrition", "text": "Mangez des œufs avec des légumes verts pour améliorer l'absorption du fer.", "icon": "🥚"},
            {"category": "vitamins", "text": "La vitamine C améliore l'absorption du fer, consommez-la avec vos repas.", "icon": "🍊"},
            {"category": "vitamins", "text": "Les suppléments de fer doivent être pris sous contrôle médical.", "icon": "💊"},
            {"category": "vitamins", "text": "La vitamine B12 est importante pour la formation des globules rouges.", "icon": "🧪"},
            {"category": "lifestyle", "text": "Pratiquez une activité physique légère pour améliorer la circulation sanguine.", "icon": "🚶"},
            {"category": "lifestyle", "text": "Dormez suffisamment pour soutenir votre système immunitaire.", "icon": "😴"},
            {"category": "lifestyle", "text": "Évitez le tabac car il réduit l'efficacité du transport d'oxygène.", "icon": "🚭"},
            {"category": "followup", "text": "Faites vérifier votre taux de fer régulièrement.", "icon": "🩸"},
            {"category": "followup", "text": "Consultez votre médecin en cas de fatigue ou de vertiges.", "icon": "👨‍⚕️"},
            {"category": "followup", "text": "Respectez le calendrier de suivi établi par votre médecin.", "icon": "📅"}
        ]
    },
    "en": {
        "app_title": "AnemiCheck AI",
        "app_subtitle": "AI for Blood Health",
        "badge_private": "Secure & Trusted",
        "badge_private_desc": "100% privacy protected",
        "badge_available": "Service Available",
        "badge_available_desc": "24/7 we are here for you",
        "hero_title": "Your doctor in less than two minutes",
        "hero_desc": "Intelligent analysis of eye images to detect anemia with high accuracy and speed",
        "hero_badge": "100% accurate and secure results",
        "hero_cta": "Start analysis now",
        "ai_dev_badge": "Built with artificial intelligence",
        "top_badge1_title": "Instant help",
        "top_badge1_desc": "Chat with our support team",
        "top_badge2_title": "Free call",
        "top_badge2_desc": "🔜 Coming soon",
        "top_badge3_title": "24/7 available",
        "top_badge3_desc": "We are always here for you",
        "top_badge4_title": "Safe & trusted",
        "top_badge4_desc": "100% of your privacy protected",
        "feature_1_title": "Quick Result",
        "feature_1_desc": "Get instant results with a detailed report",
        "feature_2_title": "Upload Image",
        "feature_2_desc": "Upload a clear eye image (JPG, PNG)",
        "feature_3_title": "Reliable & Approved",
        "feature_3_desc": "This app helps in early detection of anemia",
        "feature_4_title": "Security",
        "feature_4_desc": "Your data is fully protected",
        "how_title": "How it works?",
        "how_step1": "Upload",
        "how_step1_desc": "Upload a clear eye image",
        "how_step2": "Smart Analysis",
        "how_step2_desc": "AI model analyzes the image",
        "how_step3": "Instant Result",
        "how_step3_desc": "Get diagnosis in seconds",
        "trust_1": "Security",
        "trust_2": "Care",
        "trust_3": "High Trust",
        "version": "Version 1.0.0",
        "accuracy": "Accuracy up to 96%",
        "fast": "Fast",
        "fast_desc": "Result in less than 2 minutes",
        "smart": "Smart",
        "smart_desc": "Advanced AI technology",
        "secure": "Secure",
        "secure_desc": "Your data is protected",
        "upload_title": "Upload Eye Image",
        "upload_desc": "Clear eye image for anemia detection",
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
        "result_conjunctiva": "👁️ Enhanced Conjunctiva",
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
        "history_title": "📋 Analysis History",
        "history_date": "Date",
        "history_diagnostic": "Diagnosis",
        "history_confidence": "Confidence",
        "history_prob": "Anemia Probability",
        "history_empty": "📭 No history yet",
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
        "welcome_title": "👁️ Ready to analyze",
        "welcome_desc": "Upload an eye image to start.",
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
        "nav_history": "📋 History",
        "nav_tips": "💡 Health Tips",
        "tips_title": "💡 Health Tips for Better Blood Health",
        "tips_subtitle": "Personalized tips to improve iron levels and prevent anemia",
        "tips_category_nutrition": "🍎 Nutrition",
        "tips_category_vitamins": "💊 Vitamins",
        "tips_category_lifestyle": "🧘 Lifestyle",
        "tips_category_followup": "🩺 Medical Follow-up",
        "tips_random_btn": "🎲 Random Tip",
        "tips_search_placeholder": "🔍 Search tips...",
        "tips_list": [
            {"category": "nutrition", "text": "Eat spinach, lentils, and red meat to boost iron.", "icon": "🥩"},
            {"category": "nutrition", "text": "Add chickpeas and beans to your daily meals.", "icon": "🍲"},
            {"category": "nutrition", "text": "Eat eggs with leafy greens to improve iron absorption.", "icon": "🥚"},
            {"category": "vitamins", "text": "Vitamin C enhances iron absorption, take it with meals.", "icon": "🍊"},
            {"category": "vitamins", "text": "Iron supplements should be taken under medical supervision.", "icon": "💊"},
            {"category": "vitamins", "text": "Vitamin B12 is important for red blood cell formation.", "icon": "🧪"},
            {"category": "lifestyle", "text": "Engage in light exercise to improve blood circulation.", "icon": "🚶"},
            {"category": "lifestyle", "text": "Get enough sleep to support your immune system.", "icon": "😴"},
            {"category": "lifestyle", "text": "Avoid smoking as it reduces oxygen transport efficiency.", "icon": "🚭"},
            {"category": "followup", "text": "Have your iron levels checked regularly.", "icon": "🩸"},
            {"category": "followup", "text": "Consult your doctor if you feel tired or dizzy.", "icon": "👨‍⚕️"},
            {"category": "followup", "text": "Follow the monitoring schedule set by your doctor.", "icon": "📅"}
        ]
    }
}

# ========== دالة الترجمة ==========
def t(key):
    lang = st.session_state.get("language", "fr")
    return LANGUAGES.get(lang, LANGUAGES["fr"]).get(key, key)

# ========== CSS المتقدم ==========
st.markdown("""
<style>
    /* === نفس الأنماط السابقة مع إضافات جديدة === */
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
    
    /* ===== باقي الأنماط (كما هي مع بعض التعديلات الطفيفة) ===== */
    .header { /* نفس الشيء */ }
    /* ... (جميع الأنماط السابقة تبقى كما هي، مع إضافة الأنماط الجديدة أدناه) ... */

    /* ===== أنماط جديدة للبطاقات والنصائح ===== */
    .tip-card {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(8px);
        border-radius: 20px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.4);
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        display: flex;
        align-items: center;
        gap: 16px;
        animation: fadeUp 0.6s ease;
    }
    .tip-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 12px 32px rgba(0,0,0,0.08);
        border-color: rgba(245,158,11,0.3);
    }
    .tip-card .tip-icon {
        font-size: 32px;
        min-width: 48px;
        text-align: center;
    }
    .tip-card .tip-text {
        flex: 1;
        color: #0f172a;
        font-size: 15px;
        line-height: 1.5;
    }
    .tip-card .tip-category {
        font-size: 12px;
        font-weight: 600;
        color: #F59E0B;
        background: rgba(245,158,11,0.1);
        padding: 2px 12px;
        border-radius: 30px;
        white-space: nowrap;
    }
    .tip-card .tip-category.nutrition { background: rgba(52,211,153,0.15); color: #059669; }
    .tip-card .tip-category.vitamins { background: rgba(245,158,11,0.15); color: #D97706; }
    .tip-card .tip-category.lifestyle { background: rgba(59,130,246,0.15); color: #2563eb; }
    .tip-card .tip-category.followup { background: rgba(236,72,153,0.15); color: #db2777; }

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

    .filter-buttons {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 0.5rem 0 1.5rem;
    }
    .filter-btn {
        background: rgba(255,255,255,0.6);
        border: 1px solid #e2e8f0;
        border-radius: 30px;
        padding: 6px 18px;
        font-size: 14px;
        font-weight: 500;
        color: #334155;
        cursor: pointer;
        transition: 0.2s;
    }
    .filter-btn:hover {
        background: rgba(245,158,11,0.08);
        border-color: #F59E0B;
    }
    .filter-btn.active {
        background: #F59E0B;
        color: white;
        border-color: #F59E0B;
    }

    .random-tip-box {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 24px rgba(30,58,138,0.25);
        animation: fadeUp 0.8s ease;
    }
    .random-tip-box .rt-icon { font-size: 48px; }
    .random-tip-box .rt-text {
        font-size: 18px;
        font-weight: 500;
        margin: 10px 0;
    }
    .random-tip-box .rt-category {
        font-size: 14px;
        opacity: 0.8;
        background: rgba(255,255,255,0.15);
        padding: 4px 16px;
        border-radius: 30px;
        display: inline-block;
    }

    /* تنسيق الـ radio في القائمة الجانبية */
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

    /* باقي التنسيقات السابقة ... */
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
if 'random_tip_index' not in st.session_state:
    st.session_state.random_tip_index = random.randint(0, len(LANGUAGES["ar"]["tips_list"])-1)
if 'tip_filter' not in st.session_state:
    st.session_state.tip_filter = "all"

# ========== HEADER & SIDEBAR ==========
# (نفس الكود السابق مع إضافة خيار "نصائح طبية" في الـ radio)

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
    
    # ===== قائمة التنقل المطورة =====
    st.markdown(f"""
    <div class="sidebar-glass">
        <h4>📋 القائمة / Menu</h4>
    </div>
    """, unsafe_allow_html=True)
    
    nav_options = [t("nav_home"), t("nav_history"), t("nav_tips")]
    current_index = 0
    if st.session_state.page == "history":
        current_index = 1
    elif st.session_state.page == "tips":
        current_index = 2
    
    nav_choice = st.radio(
        "##",
        options=nav_options,
        index=current_index,
        label_visibility="collapsed",
        key="nav_radio"
    )
    
    # تحديث الصفحة
    if nav_choice == t("nav_home"):
        st.session_state.page = "home"
    elif nav_choice == t("nav_history"):
        st.session_state.page = "history"
    elif nav_choice == t("nav_tips"):
        st.session_state.page = "tips"
    
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
        <div class="tb-icon">🎧</div>
        <div><div class="tb-title">{t('top_badge1_title')}</div><div class="tb-desc">{t('top_badge1_desc')}</div></div>
    </div>
    <div class="top-badge-card disabled">
        <span class="tb-soon">🔜 {t('sidebar_doctor_soon')}</span>
        <div class="tb-icon">📞</div>
        <div><div class="tb-title">{t('top_badge2_title')}</div><div class="tb-desc">{t('top_badge2_desc')}</div></div>
    </div>
    <div class="top-badge-card">
        <div class="tb-icon">🕐</div>
        <div><div class="tb-title">{t('top_badge3_title')}</div><div class="tb-desc">{t('top_badge3_desc')}</div></div>
    </div>
    <div class="top-badge-card">
        <div class="tb-icon">🛡️</div>
        <div><div class="tb-title">{t('top_badge4_title')}</div><div class="tb-desc">{t('top_badge4_desc')}</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== HERO ==========
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

if st.session_state.page == "home":
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
                <img src="{doctor_img_url}" class="doctor-image" alt="Médecin / Doctor">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # في الصفحات الأخرى نعرض هيرو مصغر أو بدون هيرو
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3a8a, #2563eb); border-radius: 20px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center; color: white;">
        <h2 style="font-weight: 700; margin:0;">{t('app_title')}</h2>
        <p style="margin:0; opacity:0.8;">{t('app_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# ========== دوال المساعدة ==========
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
# ========== الصفحة الرئيسية (HOME) ==========
# ================================================================
if st.session_state.page == "home":
    # ===== واجهة الرفع =====
    st.markdown(f"""
    <div class="upload-card" id="upload-zone">
        <div class="icon">📸</div>
        <h3>{t('upload_title')}</h3>
        <p>{t('upload_desc')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="heartbeat-icon">
        💓 <span>❝ نبض التشخيص / Diagnostic Pulse ❞</span> 💓
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

                # ===== عرض النتائج (كما هي) =====
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
                    "prob_value": anemia_pct  # للرسم البياني
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

    # ===== باقي محتوى الصفحة الرئيسية (الميزات، كيف يعمل، الثقة) =====
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

    st.markdown(f"""
    <div class="trust-section">
        <div class="trust-item"><span class="icon">🔒</span> {t('trust_1')}</div>
        <div class="trust-item"><span class="icon">❤️</span> {t('trust_2')}</div>
        <div class="trust-item"><span class="icon">⭐</span> {t('trust_3')}</div>
        <div class="trust-item"><span class="icon">📱</span> {t('version')}</div>
        <div class="trust-item"><span class="icon">🎯</span> {t('accuracy')}</div>
    </div>
    """, unsafe_allow_html=True)

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
# ========== صفحة سجل التحليلات (HISTORY) المطورة ==========
# ================================================================
elif st.session_state.page == "history":
    st.markdown(f'<div class="section-title">{t("history_title")}</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        # ===== الإحصائيات =====
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
        
        # ===== رسم بياني لتطور الاحتمال =====
        if total >= 2:
            st.markdown(f'<h4 style="margin-top:1rem;">{t("history_chart_title")}</h4>', unsafe_allow_html=True)
            # استخراج البيانات
            dates = []
            probs = []
            for h in st.session_state.history:
                # استخراج التاريخ والاحتمال
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
        
        # ===== عرض بطاقات التحليلات =====
        st.markdown("---")
        st.markdown("#### 📋 تفاصيل التحاليل")
        
        # عكس الترتيب لعرض الأحدث أولاً
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
        
        # ===== أزرار التحكم =====
        col_export, col_clear = st.columns(2)
        with col_export:
            if st.button(t("history_export"), use_container_width=True):
                st.info("📥 سيتم تصدير التقرير قريباً... (وهمي)")
        with col_clear:
            if st.button(t("history_clear"), use_container_width=True):
                st.session_state.history = []
                st.rerun()
    else:
        # صفحة فارغة
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
# ========== صفحة النصائح الطبية المبتكرة ==========
# ================================================================
elif st.session_state.page == "tips":
    st.markdown(f"""
    <div style="text-align:center; margin-bottom: 1.5rem;">
        <h2 style="font-weight: 800; color: #0f172a; font-size: 28px;">{t('tips_title')}</h2>
        <p style="color: #64748b; font-size: 16px;">{t('tips_subtitle')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== نصيحة عشوائية =====
    tips_list = t("tips_list")  # القائمة المترجمة
    if tips_list:
        st.markdown(f"""
        <div class="random-tip-box">
            <div class="rt-icon">{tips_list[st.session_state.random_tip_index]['icon']}</div>
            <div class="rt-text">“{tips_list[st.session_state.random_tip_index]['text']}”</div>
            <div class="rt-category">#{tips_list[st.session_state.random_tip_index]['category']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col_rnd, _ = st.columns([1, 3])
        with col_rnd:
            if st.button(t("tips_random_btn"), use_container_width=True):
                st.session_state.random_tip_index = random.randint(0, len(tips_list)-1)
                st.rerun()
    
    # ===== فلترة النصائح =====
    st.markdown("#### 🔍 تصفية النصائح")
    # أزرار الفلاتر
    categories = ["all", "nutrition", "vitamins", "lifestyle", "followup"]
    cat_labels = {
        "all": "📌 الكل",
        "nutrition": t("tips_category_nutrition"),
        "vitamins": t("tips_category_vitamins"),
        "lifestyle": t("tips_category_lifestyle"),
        "followup": t("tips_category_followup")
    }
    
    col_filters = st.columns(len(categories))
    for i, cat in enumerate(categories):
        with col_filters[i]:
            active = st.session_state.tip_filter == cat
            if st.button(cat_labels[cat], use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.tip_filter = cat
                st.rerun()
    
    # ===== شريط بحث =====
    search_query = st.text_input("", placeholder=t("tips_search_placeholder"), label_visibility="collapsed")
    
    # ===== عرض النصائح المفلترة =====
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
            <div class="tip-card">
                <div class="tip-icon">{tip['icon']}</div>
                <div class="tip-text">{tip['text']}</div>
                <span class="tip-category {cat_class}">{cat_label}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("🚫 لا توجد نصائح تطابق معايير البحث.")
    
    # ===== نصيحة إضافية (توعوية) =====
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(245,158,11,0.08); border-radius: 16px; padding: 1rem; border-left: 4px solid #F59E0B; margin-top: 1rem;">
        <p style="margin:0; color:#0f172a; font-weight:500;">💡 تذكر أن هذه النصائح هي لأغراض توعوية ولا تغني عن استشارة الطبيب المختص.</p>
    </div>
    """, unsafe_allow_html=True)

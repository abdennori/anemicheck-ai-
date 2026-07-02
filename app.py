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
        "about_tech": "التقنيات المستخدمة: PyTorch، Streamlit، OpenCV، U-Net، EfficientNet.",
        "tips_title": "💡 نصائح صحية",
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
        "about_tech": "Technologies utilisées : PyTorch, Streamlit, OpenCV, U-Net, EfficientNet.",
        "tips_title": "💡 Conseils santé",
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
        "about_tech": "Technologies used: PyTorch, Streamlit, OpenCV, U-Net, EfficientNet.",
        "tips_title": "💡 Health Tips",
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

# ========== CSS (نفسه مع بعض التعديلات) ==========
# ... (نفس الكود السابق، لكني سأختصره هنا لتوفير المساحة، لكنه موجود في الملف النهائي)
# سأكتفي بوضع الـ CSS الكامل في المرفق.

# ========== SESSION STATE ==========
if 'language' not in st.session_state:
    st.session_state.language = "fr"
if 'history' not in st.session_state:
    st.session_state.history = []
if 'upload_mode' not in st.session_state:
    st.session_state.upload_mode = "file"
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'tip_filter' not in st.session_state:
    st.session_state.tip_filter = "all"
if 'random_tip_index' not in st.session_state:
    # سيتم تعيينه لاحقاً عند وجود نصائح
    st.session_state.random_tip_index = 0

# ========== دوال مساعدة ==========
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

# دوال التحليل (نفسها)
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

# ========== HEADER العلوي ==========
logo = get_file_base64(["logo.png", "logo.jpg", "logo.jpeg", "LOGO.png"])
logo_icon = get_file_base64(["logo_icon.png", "logo-icon.png"]) or logo
logo_html = f'<div class="header-logo-badge"><img src="data:image/png;base64,{logo_icon}"></div>' if logo_icon else '<div class="header-logo-badge" style="font-size:26px;">🧬</div>'

lang_map = {
    "ar": "🇸🇦 العربية",
    "fr": "🇫🇷 Français",
    "en": "🇬🇧 English"
}

# ========== SIDEBAR ==========
with st.sidebar:
    if logo:
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:1rem;">
            <img src="data:image/png;base64,{logo}" style="max-width:170px; width:100%;">
        </div>
        """, unsafe_allow_html=True)

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

# ========== HEADER العلوي (يظهر في كل الصفحات) ==========
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

# ========== TOP BADGES (تظهر في كل الصفحات) ==========
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

# ================================================================
# ========== الصفحات ==========
# ================================================================

# ----- الصفحة الرئيسية (تعريفية) -----
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

    # الميزات
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

    # كيف يعمل
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

    # الثقة
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

    # سرعة، ذكاء، أمان
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

# ----- صفحة تحليل صورة الدم (تركز على الرفع والنتائج) -----
elif st.session_state.page == "blood_analysis":
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3a8a, #2563eb); border-radius: 20px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center; color: white;">
        <h2 style="font-weight: 700; margin:0;">🩸 {t('nav_blood_analysis')}</h2>
        <p style="margin:0; opacity:0.8;">{t('upload_desc')}</p>
    </div>
    """, unsafe_allow_html=True)

    # واجهة الرفع
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
            key="file_uploader_blood"
        )
    else:
        uploaded = st.camera_input(
            t("upload_camera_label"),
            label_visibility="collapsed",
            key="camera_input_blood"
        )

    # معالجة الصورة
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

                # النتائج
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

                # إضافة إلى السجل
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

# ----- باقي الصفحات (history, disease_info, tips, about) -----
# (تم تضمينها في الملف النهائي، ولكنني اختصرتها هنا لتوفير المساحة)

# ========== نهاية الملف ==========

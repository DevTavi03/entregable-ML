import streamlit as st
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA (ESTILO APPLE)
# ==========================================
st.set_page_config(page_title="VisionOCR | TecnoForms", layout="wide", initial_sidebar_state="collapsed")

# Inyección de CSS (UI/UX Apple Style, Glassmorphism, San Francisco Font)
st.markdown("""
<style>
    /* Tipografía y Fondo General */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
        background-color: #F5F5F7;
        color: #1D1D1F;
    }
    
    /* Ocultar elementos por defecto de Streamlit (Header, Footer, Menú) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Estilo del contenedor principal (Efecto Tarjeta/Glass) */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Títulos limpios */
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.5px !important;
    }

    /* Estilo de la zona de subida de archivos (Dropzone) */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 2px dashed #D2D2D7;
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1);
    }
    .stFileUploader > div > div:hover {
        border-color: #0066CC;
        box-shadow: 0 8px 24px rgba(0, 102, 204, 0.15);
        transform: scale(1.01);
    }

    /* Botones estilo Apple */
    .stButton > button {
        background-color: #0066CC;
        color: white;
        border-radius: 980px; /* Pill shape */
        padding: 0.5rem 2rem;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #0055AA;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
        transform: translateY(-1px);
    }
    
    /* Panel de resultados (Tarjetas Blancas) */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ENCABEZADO MINIMALISTA
# ==========================================
col_logo, col_status = st.columns([4, 1])
with col_logo:
    st.markdown("<h1> VisionOCR <span style='font-size: 20px; font-weight: 400; color: #86868B;'>by TecnoForms</span></h1>", unsafe_allow_html=True)
with col_status:
    st.markdown("<div style='text-align: right; margin-top: 20px; font-weight: 600; color: #34C759;'>● Backend Conectado</div>", unsafe_allow_html=True)

st.markdown("<p style='color: #86868B; font-size: 1.1rem; margin-bottom: 2rem;'>Inteligencia Artificial para reconocimiento de documentos y formularios manuscritos.</p>", unsafe_allow_html=True)

# ==========================================
# 3. CARGA DEL MODELO IA (En Caché para no recargar)
# ==========================================
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    # En Streamlit Cloud usamos gpu=False porque no hay GPU gratuita
    return easyocr.Reader(['es', 'en'], gpu=False)

modelo = cargar_modelo()

# ==========================================
# 4. ZONA DE CARGA (DROPZONE)
# ==========================================
archivo_subido = st.file_uploader("Arrastra una imagen aquí o selecciona un archivo", type=["png", "jpg", "jpeg"])

# ==========================================
# 5. PROCESAMIENTO Y PANEL DE RESULTADOS
# ==========================================
if archivo_subido is not None:
    # Mostrar estado de carga (Spinner minimalista)
    with st.spinner('Analizando documento con Deep Learning...'):
        
        # Leer la imagen
        imagen_bytes = archivo_subido.read()
        imagen_pil = Image.open(io.BytesIO(imagen_bytes)).convert('RGB')
        imagen_np = np.array(imagen_pil)
        
        # Ejecutar OCR (Mag_ratio y contrast ajustados para manuscritos)
        resultados = modelo.readtext(imagen_np, detail=1, mag_ratio=1.5, contrast_ths=0.1, adjust_contrast=0.5)
        
        # Preparar para dibujar sobre la imagen
        draw = ImageDraw.Draw(imagen_pil)
        try:
            # Intentar cargar una fuente del sistema, si no, usa default
            fuente = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            fuente = ImageFont.load_default()

        textos_extraidos = []
        
        # Procesar cada bloque de texto detectado
        for (coordenadas, texto, confianza) in resultados:
            (top_left, top_right, bottom_right, bottom_left) = coordenadas
            p0 = (int(top_left[0]), int(top_left[1]))
            p2 = (int(bottom_right[0]), int(bottom_right[1]))
            
            texto_limpio = texto.upper()
            textos_extraidos.append(texto_limpio)
            
            # Dibujar caja y texto (Estilo profesional: Caja azul claro)
            draw.rectangle([p0, p2], outline="#0066CC", width=3)
            
            # Fondo blanco para el texto para que resalte
            bbox = draw.textbbox((p0[0], p0[1] - 25), texto_limpio, font=fuente)
            draw.rectangle(bbox, fill="white")
            draw.text((p0[0], p0[1] - 25), texto_limpio, font=fuente, fill="#0066CC")
            
    # ==========================================
    # 6. SPLIT SCREEN (RESULTADOS)
    # ==========================================
    st.markdown("<hr style='border: 1px solid #E5E5EA;'>", unsafe_allow_html=True)
    st.markdown("<h3>Resultados del Análisis</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("<p style='font-weight:600;'>Imagen Procesada (Layout Analysis)</p>", unsafe_allow_html=True)
        # Mostramos la imagen con las cajas dibujadas
        st.image(imagen_pil, use_column_width=True, clamp=True)
        
    with col2:
        st.markdown("<p style='font-weight:600;'>Texto Extraído</p>", unsafe_allow_html=True)
        
        if textos_extraidos:
            texto_final = "\n".join(textos_extraidos)
            
            # Usamos st.code que incluye nativamente el botón de "Copiar al portapapeles"
            # Es limpio, minimalista y cumple la función perfectamente
            st.code(texto_final, language='text')
            
            st.success("Extracción completada. El texto está listo para copiar.")
            
            # Mostrar tabla de confianza (Valor agregado)
            st.markdown("<p style='font-weight:600; margin-top: 20px;'>Detalle de Confianza</p>", unsafe_allow_html=True)
            datos_confianza = [{"Texto": t.upper(), "Confianza": f"{c*100:.1f}%"} for (_, t, c) in resultados]
            st.dataframe(datos_confianza, use_container_width=True)
        else:
            st.warning("No se detectó texto en la imagen.")
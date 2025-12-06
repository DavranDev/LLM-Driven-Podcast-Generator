"""
Configuration file for Campus Copilot Podcast Generator
Customize settings here instead of modifying the code directly
"""

# =============================================================================
# API CONFIGURATION
# =============================================================================

# LLM Configuration
LLM_PROVIDER = "openai"  # Options: "openai", "groq"

# OpenAI API Configuration
OPENAI_API_KEY = None  # Set via environment variable or here
OPENAI_MODEL_RESEARCH = "gpt-4o"  # For content analysis
OPENAI_MODEL_SCRIPT = "gpt-4o"  # For creative dialogue

# Groq API Configuration
GROQ_API_KEY = None  # Set via environment variable or here
GROQ_MODEL = "llama-3.3-70b-versatile"  # Groq's Llama 3.3 70B model

# TTS Configuration
TTS_PROVIDER = "openai"  # Only OpenAI TTS is supported
TTS_MODEL = "tts-1-hd"  # Options: "tts-1", "tts-1-hd"

# =============================================================================
# PHASE 1: RESEARCHER SETTINGS
# =============================================================================

# Gap Analysis
MAX_KNOWLEDGE_GAPS = 10  # Maximum gaps to identify and fill
GAP_ANALYSIS_TEMPERATURE = 0.3  # Lower = more focused, higher = more creative

# Web Search
MAX_SEARCH_RESULTS = 3  # Results per gap search
SEARCH_TIMEOUT = 10  # Seconds

# Content Processing
MAX_CONTENT_LENGTH = 10000  # Characters before chunking
CHUNK_OVERLAP = 500  # Character overlap between chunks

# =============================================================================
# PHASE 2: SCRIPTWRITER SETTINGS
# =============================================================================

# Script Generation
SCRIPT_TEMPERATURE = 0.85  # Higher = more creative/varied dialogue
MAX_SEGMENTS = 7  # Maximum content segments in podcast
TARGET_WORDS_PER_SEGMENT = 350  # Reduced for more concise segments

# Host Personas
HOST_A_NAME = "Alexa"
HOST_A_ROLE = "The Curious Student"
HOST_A_VOICE = "nova"  # Female, warm (NotebookLM-style)

HOST_B_NAME = "Jordan"
HOST_B_ROLE = "The Knowledgeable Guide"
HOST_B_VOICE = "onyx"  # Male, deep (NotebookLM-style)

# Available OpenAI TTS Voices:
# RECOMMENDED FOR NOTEBOOKLM-STYLE:
# - nova: Female, warm, conversational (Best for curious student)
# - onyx: Male, deep, authoritative (Best for expert guide)
#
# OTHER OPTIONS:
# - alloy: Neutral, balanced
# - echo: Male, clear
# - fable: Male, expressive
# - shimmer: Female, bright

# Dialogue Realism Settings
USE_FILLER_WORDS = True
USE_INTERRUPTIONS = True
USE_OVERLAPS = True
USE_EMOTIONAL_MARKERS = True
USE_BACKCHANNELS = True  # "mhm", "right", "yeah" from listener

INTERRUPTION_FREQUENCY = "high"  # Options: "low", "medium", "high"
OVERLAP_FREQUENCY = "medium"  # Options: "low", "medium", "high"
BACKCHANNEL_FREQUENCY = "high"  # How often the listener gives feedback

# Response Length Control
MAX_WORDS_PER_RESPONSE = 120  # Keep individual responses concise
PREFER_CONVERSATIONAL_CHUNKS = True  # Break up long explanations

# =============================================================================
# PHASE 3: AUDIO PRODUCER SETTINGS
# =============================================================================

# Voice Settings
VOICE_SPEEDS = {
    'HOST_A': 0.95,   # Normal speed for curious student
    'HOST_B': 0.86   # Slightly slower for authority figure
}

# Timing Settings (in seconds)
DEFAULT_PAUSE = 0.25  # Default pause between lines (reduced for natural flow)
SHORT_PAUSE = 0.1   # [BEAT] marker
LONG_PAUSE = 1.5    # [PAUSE] marker
INTERRUPT_OFFSET = 0.2  # How early interruptions start
OVERLAP_OFFSET = 0.25    # How much overlaps overlap

# Audio Quality
AUDIO_BITRATE = "256k"  # MP3 bitrate (128k, 192k, 256k, 320k)
SAMPLE_RATE = 44100  # Hz (professional podcast standard)
TARGET_LOUDNESS = -16  # dBFS for final audio (broadcast standard)
COMPRESSION_RATIO = 2.5  # Dynamic range compression (gentler for natural sound)

# Audio Effects
ADD_ROOM_TONE = False  # Add subtle ambient sound (NotebookLM uses minimal/none)
ROOM_TONE_VOLUME = -50  # dB (extremely quiet, barely perceptible)
NORMALIZE_AUDIO = True
APPLY_COMPRESSION = True
VOLUME_DUCK_ON_OVERLAP = -4  # dB reduction during overlaps (gentle)

# Caching
USE_TTS_CACHE = True  # Cache TTS clips to save API costs
CACHE_DIRECTORY = "data/outputs/audio_cache"
DELETE_CACHE_AFTER_SUCCESS = False  # Delete cache after successful podcast generation

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

# Directory Structure
OUTPUT_DIR = "./data/outputs"
ENRICHED_DIR = "enriched_content"
SCRIPTS_DIR = "scripts"
AUDIO_DIR = "audio"

# File Naming
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
USE_DESCRIPTIVE_NAMES = True  # Use title in filenames

# Export Formats
EXPORT_SCRIPT_TXT = True  # Human-readable script
EXPORT_SCRIPT_JSON = True  # Machine-readable for audio
EXPORT_ENRICHED_MD = True  # Enriched content as markdown
SAVE_RESULTS_SUMMARY = True  # JSON summary of outputs

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Performance
MAX_CONCURRENT_TTS = 1  # Parallel TTS requests (be careful with rate limits)
TTS_RATE_LIMIT_DELAY = 0.5  # Seconds between TTS calls

# Debugging
VERBOSE_LOGGING = True
SAVE_INTERMEDIATE_FILES = True  # Keep all intermediate processing files
LOG_API_CALLS = False  # Log all API requests (for debugging)

# Error Handling
MAX_RETRIES = 3  # For API calls
RETRY_DELAY = 5  # Seconds between retries
FALLBACK_ON_ERROR = True  # Continue with partial results on errors

# =============================================================================
# EXPERIMENTAL FEATURES
# =============================================================================

# Future features (not yet implemented)
USE_BACKGROUND_MUSIC = False
BACKGROUND_MUSIC_VOLUME = -30  # dB

USE_SOUND_EFFECTS = False  # Add subtle sound effects

MULTI_LANGUAGE = False
TARGET_LANGUAGE = "en"  # ISO language code

# =============================================================================
# QUALITY PRESETS
# =============================================================================

QUALITY_PRESETS = {
    'draft': {
        'OPENAI_MODEL_SCRIPT': 'gpt-4o',
        'TTS_MODEL': 'tts-1',
        'AUDIO_BITRATE': '128k',
        'MAX_KNOWLEDGE_GAPS': 5
    },
    'standard': {
        'OPENAI_MODEL_SCRIPT': 'gpt-4o',
        'TTS_MODEL': 'tts-1-hd',
        'AUDIO_BITRATE': '192k',
        'MAX_KNOWLEDGE_GAPS': 10
    },
    'premium': {
        'OPENAI_MODEL_SCRIPT': 'gpt-4o',
        'TTS_MODEL': 'tts-1-hd',
        'AUDIO_BITRATE': '320k',
        'MAX_KNOWLEDGE_GAPS': 15
    }
}

# Active preset
ACTIVE_PRESET = 'premium'  # Options: 'draft', 'standard', 'premium'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_preset(preset_name):
    """Load a quality preset"""
    if preset_name in QUALITY_PRESETS:
        preset = QUALITY_PRESETS[preset_name]
        globals().update(preset)
        print(f"Loaded preset: {preset_name}")
    else:
        print(f"Unknown preset: {preset_name}")

def get_config():
    """Get current configuration as dictionary"""
    return {k: v for k, v in globals().items() if k.isupper()}

# Auto-load active preset on import
if ACTIVE_PRESET:
    load_preset(ACTIVE_PRESET)
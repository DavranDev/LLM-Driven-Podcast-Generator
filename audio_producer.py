"""
Audio Producer Agent - Phase 3 (ENHANCED WITH NOISE REDUCTION)
NotebookLM-inspired high-quality audio with clean mixing

ENHANCEMENTS:
- High-pass filter (80Hz): Removes low-frequency rumble/hum
- Low-pass filter (12kHz): Removes high-frequency hiss  
- Noise gate: Silences background noise between words
- Subtle fades: Prevents clicks and pops at audio boundaries
- Improved normalization and compression
"""

import os
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib
import shutil
import io

# Audio processing
from pydub import AudioSegment
from pydub.effects import (
    normalize, 
    compress_dynamic_range,
    high_pass_filter,
    low_pass_filter
)
from pydub.silence import detect_leading_silence

# OpenAI TTS
from openai import OpenAI


@dataclass
class AudioClip:
    """Represents a single audio clip with metadata"""
    speaker: str
    text: str
    audio: AudioSegment
    emotion: Optional[str] = None
    action: Optional[str] = None
    duration_ms: int = 0
    
    def __post_init__(self):
        if self.audio:
            self.duration_ms = len(self.audio)


@dataclass 
class TimedAudioClip:
    """Audio clip with precise timing information"""
    clip: AudioClip
    start_time_ms: int
    end_time_ms: int
    volume_adjustment: float = 1.0


class TTSEngine:
    """OpenAI TTS engine with professional noise reduction"""
    
    def __init__(self, openai_api_key: str, model: str = "tts-1-hd"):
        """
        Initialize OpenAI TTS engine
        
        Args:
            openai_api_key: OpenAI API key
            model: TTS model ("tts-1" or "tts-1-hd")
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        
        # NotebookLM-inspired voice pairing
        self.voice_map = {
            'HOST_A': 'nova',   # Female, warm, engaging
            'HOST_B': 'onyx'   # Male, deep, authoritative
        }
        
        # Voice speed settings
        self.speed_map = {
            'HOST_A': 0.96,  # Slightly faster - enthusiastic
            'HOST_B': 0.85   # Slightly slower - thoughtful
        }
        
        self.noise_reduction_enabled = True
        self.high_pass_cutoff = 80      # Hz - removes rumble
        self.low_pass_cutoff = 12000    # Hz - removes hiss
        self.noise_gate_threshold = -45  # dB - below this is "noise"
        self.noise_gate_reduction = 20   # dB - how much to reduce noise
        
        print(f"OpenAI TTS engine initialized (model: {model})")
        print(f"   Voices: nova (female/curious) + onyx (male/expert)")
        print(f"Noise reduction: ENABLED")
    
    def configure_voices(self, host_a_voice: str = 'nova', host_b_voice: str = 'onyx',
                         host_a_speed: float = 0.96, host_b_speed: float = 0.85):
        """
        Configure voices for both hosts
        
        Available voices:
        - alloy: Neutral, balanced
        - echo: Male, clear
        - fable: Male, expressive
        - onyx: Male, deep
        - nova: Female, warm
        - shimmer: Female, bright
        """
        self.voice_map['HOST_A'] = host_a_voice
        self.voice_map['HOST_B'] = host_b_voice
        self.speed_map['HOST_A'] = host_a_speed
        self.speed_map['HOST_B'] = host_b_speed
        print(f"🎤 Voices configured: HOST_A={host_a_voice}, HOST_B={host_b_voice}")
    
    def configure_noise_reduction(self, 
                                   enabled: bool = True,
                                   high_pass_hz: int = 80,
                                   low_pass_hz: int = 12000,
                                   gate_threshold_db: float = -45,
                                   gate_reduction_db: float = 20):
        """
        Configure noise reduction settings
        
        Args:
            enabled: Enable/disable noise reduction
            high_pass_hz: High-pass filter cutoff (removes rumble below this)
            low_pass_hz: Low-pass filter cutoff (removes hiss above this)
            gate_threshold_db: Audio below this level is considered noise
            gate_reduction_db: How much to reduce detected noise
        """
        self.noise_reduction_enabled = enabled
        self.high_pass_cutoff = high_pass_hz
        self.low_pass_cutoff = low_pass_hz
        self.noise_gate_threshold = gate_threshold_db
        self.noise_gate_reduction = gate_reduction_db
        
        status = "ENABLED" if enabled else "DISABLED"
        print(f"🔇 Noise reduction: {status}")
        if enabled:
            print(f"   High-pass: {high_pass_hz}Hz | Low-pass: {low_pass_hz}Hz")
            print(f"   Gate threshold: {gate_threshold_db}dB | Reduction: {gate_reduction_db}dB")
    
    def generate_speech(self, text: str, speaker: str, 
                        emotion: Optional[str] = None) -> AudioSegment:
        """
        Generate high-quality speech with professional noise reduction
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID (HOST_A or HOST_B)
            emotion: Optional emotion marker
            
        Returns:
            AudioSegment of clean, processed speech
        """
        voice = self.voice_map.get(speaker, 'nova')
        speed = self.speed_map.get(speaker, 1.0)
        
        # Adjust speed based on emotion
        if emotion == 'EXCITED':
            speed *= 1.00
        elif emotion == 'THOUGHTFUL':
            speed *= 0.78
        elif emotion == 'LAUGH':
            speed *= 0.95
        elif emotion == 'CHUCKLE':
            speed *= 0.92
        
        # Clamp speed to OpenAI's limits
        speed = max(0.25, min(4.0, speed))
        
        # Generate speech with OpenAI
        try:
            response = self.client.audio.speech.create(
                model=self.model,
                voice=voice,
                input=text,
                speed=speed
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI TTS API call failed: {e}")
        
        # Convert to AudioSegment
        audio_bytes = response.content
        
        if not audio_bytes:
            raise ValueError(f"OpenAI TTS returned empty audio for text: {text[:50]}...")
        if len(audio_bytes) < 100:
            raise ValueError(f"OpenAI TTS returned very small audio ({len(audio_bytes)} bytes)")
        
        try:
            audio = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
        except Exception as e:
            raise RuntimeError(f"Failed to convert audio bytes to AudioSegment: {e}")
        
        if audio is None or len(audio) == 0:
            raise ValueError("AudioSegment is empty after conversion")
        
        # Apply professional noise reduction and cleanup
        audio = self._clean_audio(audio)
        
        if len(audio) == 0:
            raise ValueError("Audio became empty after cleaning")
        
        return audio
    
    def _apply_noise_gate(self, audio: AudioSegment) -> AudioSegment:
        """
        Apply noise gate to silence quiet background noise
        
        This reduces the volume of very quiet sections that are
        likely just background noise, not speech.
        """
        chunk_size_ms = 50  # Process in 50ms chunks
        threshold_dB = self.noise_gate_threshold
        reduction_dB = self.noise_gate_reduction
        
        if len(audio) < chunk_size_ms:
            return audio
        
        chunks = []
        for i in range(0, len(audio), chunk_size_ms):
            chunk = audio[i:i + chunk_size_ms]
            
            # If this chunk is below threshold, it's likely noise
            if chunk.dBFS < threshold_dB:
                chunk = chunk.apply_gain(-reduction_dB)
            
            chunks.append(chunk)
        
        if not chunks:
            return audio
        
        # Reassemble the audio
        result = chunks[0]
        for chunk in chunks[1:]:
            result = result + chunk
        
        return result
    
    def _clean_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        Professional audio cleaning with noise reduction
        
        Pipeline:
        1. High-pass filter (80Hz) - Removes low-frequency rumble/hum
        2. Low-pass filter (12kHz) - Removes high-frequency hiss
        3. Noise gate - Silences very quiet noise between words
        4. Trim silence - Removes dead air at start/end
        5. Subtle fades - Prevents clicks and pops
        6. Normalization - Consistent volume levels
        7. Light compression - Even dynamics
        """
        if audio is None or len(audio) < 50:
            return audio
        
        cleaned = audio
        
        # === STEP 1: High-Pass Filter ===
        # Removes low-frequency rumble, hum, and handling noise
        if self.noise_reduction_enabled:
            try:
                cleaned = high_pass_filter(cleaned, cutoff=self.high_pass_cutoff)
            except Exception as e:
                print(f"High-pass filter failed: {e}")
        
        # === STEP 2: Low-Pass Filter ===
        # Removes high-frequency hiss and artifacts
        # TTS audio rarely has useful content above 12kHz
        if self.noise_reduction_enabled:
            try:
                cleaned = low_pass_filter(cleaned, cutoff=self.low_pass_cutoff)
            except Exception as e:
                print(f"Low-pass filter failed: {e}")
        
        # === STEP 3: Noise Gate ===
        # Silences very quiet parts that are likely background noise
        if self.noise_reduction_enabled:
            try:
                cleaned = self._apply_noise_gate(cleaned)
            except Exception as e:
                print(f"Noise gate failed: {e}")
        
        # === STEP 4: Trim Silence from Start ===
        try:
            trim_ms = detect_leading_silence(cleaned, silence_threshold=-40)
            if 0 < trim_ms < len(cleaned) - 50:
                cleaned = cleaned[trim_ms:]
        except Exception as e:
            print(f"Start trim failed: {e}")
        
        # === STEP 5: Trim Silence from End ===
        try:
            reversed_audio = cleaned.reverse()
            trim_end_ms = detect_leading_silence(reversed_audio, silence_threshold=-40)
            if 0 < trim_end_ms < len(cleaned) - 50:
                cleaned = cleaned[:-trim_end_ms]
        except Exception as e:
            print(f"End trim failed: {e}")
        
        # === STEP 6: Subtle Fades ===
        # Prevents clicks and pops at audio boundaries
        try:
            if len(cleaned) > 30:
                cleaned = cleaned.fade_in(10)   # 10ms fade in
                cleaned = cleaned.fade_out(10)  # 10ms fade out
        except Exception as e:
            print(f"Fade failed: {e}")
        
        # === STEP 7: Normalization ===
        target_dBFS = -20.0  # Standard podcast level
        try:
            if cleaned.dBFS != float('-inf') and cleaned.dBFS != 0:
                change_in_dBFS = target_dBFS - cleaned.dBFS
                change_in_dBFS = max(-20, min(20, change_in_dBFS))
                if abs(change_in_dBFS) > 0.5:
                    cleaned = cleaned.apply_gain(change_in_dBFS)
        except Exception as e:
            print(f"Normalization failed: {e}")
        
        # === STEP 8: Light Compression ===
        # Evens out volume spikes for consistent levels
        try:
            if len(cleaned) > 100:
                cleaned = compress_dynamic_range(
                    cleaned,
                    threshold=-22.0,
                    ratio=2.0,
                    attack=5.0,
                    release=50.0
                )
        except Exception as e:
            print(f"Compression failed: {e}")
        
        return cleaned


class AudioComposer:
    """Professional audio composer with clean mixing"""
    
    def __init__(self):
        pass
    
    def compose(self, timeline: 'ConversationTimeline', 
                master_compression: bool = True) -> AudioSegment:
        """
        Compose final audio with professional mixing
        
        Args:
            timeline: The conversation timeline
            master_compression: Apply final mastering compression
            
        Returns:
            Final mixed AudioSegment
        """
        print("\nComposing final audio...")
        
        if not timeline.clips:
            print("Warning: No clips to mix!")
            return AudioSegment.silent(duration=1000)
        
        total_duration = timeline.get_total_duration_ms()
        if total_duration == 0:
            print("Warning: Timeline has 0 duration!")
            return AudioSegment.silent(duration=1000)
        
        print(f"    Total duration: {total_duration}ms ({total_duration/1000:.1f}s)")
        print(f"    Processing {len(timeline.clips)} clips...")
        
        # Create silent base track
        final_audio = AudioSegment.silent(duration=total_duration)
        
        # Sort clips by start time
        sorted_clips = sorted(timeline.clips, key=lambda c: c.start_time_ms)
        
        clips_mixed = 0
        errors = 0
        
        for i, timed_clip in enumerate(sorted_clips, 1):
            try:
                clip = timed_clip.clip
                start_ms = timed_clip.start_time_ms
                
                if clip.audio is None or len(clip.audio) == 0:
                    print(f"Warning: Clip {i} is empty, skipping")
                    errors += 1
                    continue
                
                # Overlay this clip onto the final audio
                final_audio = final_audio.overlay(clip.audio, position=start_ms)
                clips_mixed += 1
                
                if i % 20 == 0:
                    print(f"    Mixed {i}/{len(sorted_clips)} clips...")
                    
            except Exception as e:
                print(f"Error mixing clip {i}: {e}")
                errors += 1
                continue
        
        print(f"Successfully mixed {clips_mixed}/{len(sorted_clips)} clips")
        if errors > 0:
            print(f"Skipped {errors} clips due to errors")
        
        # Validate
        if len(final_audio) == 0:
            raise RuntimeError("Final audio is empty!")
        
        print(f"    Final audio length: {len(final_audio)/1000:.1f}s")
        
        # Final mastering
        if master_compression:
            print("    Applying final mastering...")
            
            try:
                final_audio = compress_dynamic_range(
                    final_audio,
                    threshold=-18.0,
                    ratio=2.5,
                    attack=5.0,
                    release=50.0
                )
            except Exception as e:
                print(f"Compression failed: {e}")
            
            # Final normalization to podcast standard
            target_dBFS = -16.0
            try:
                if final_audio.dBFS != float('-inf') and final_audio.dBFS != 0:
                    change_in_dBFS = target_dBFS - final_audio.dBFS
                    change_in_dBFS = max(-20, min(20, change_in_dBFS))
                    final_audio = final_audio.apply_gain(change_in_dBFS)
                    print(f"Applied normalization: {change_in_dBFS:.1f} dB")
            except Exception as e:
                print(f"Normalization failed: {e}")
        
        print("Clean, professional mix complete")
        return final_audio


class ConversationTimeline:
    """Manages the timeline and timing of the conversation"""
    
    def __init__(self):
        self.clips: List[TimedAudioClip] = []
        self.current_time_ms = 0
    
    def add_clip(self, clip: AudioClip, timing_before: float = 0.3,
                 action: Optional[str] = None) -> int:
        """
        Add a clip to the timeline with intelligent timing
        
        Args:
            clip: The audio clip to add
            timing_before: Seconds of silence before this clip
            action: Special action (INTERRUPT, OVERLAP, etc.)
            
        Returns:
            Start time in milliseconds
        """
        if action == "INTERRUPT":
            if self.clips:
                interrupt_offset = 250
                prev_clip = self.clips[-1]
                start_time = max(0, prev_clip.end_time_ms - interrupt_offset)
            else:
                start_time = self.current_time_ms
                
        elif action == "OVERLAP-START":
            if self.clips:
                overlap_offset = 350
                prev_clip = self.clips[-1]
                start_time = max(0, prev_clip.end_time_ms - overlap_offset)
            else:
                start_time = self.current_time_ms
                
        elif action == "BEAT":
            start_time = self.current_time_ms + 150
            
        elif action == "PAUSE":
            start_time = self.current_time_ms + 1200
            
        else:
            delay_ms = int(timing_before * 1000)
            delay_ms = int(delay_ms * 0.7)  # Reduce for natural flow
            start_time = self.current_time_ms + delay_ms
        
        end_time = start_time + clip.duration_ms
        
        timed_clip = TimedAudioClip(
            clip=clip,
            start_time_ms=start_time,
            end_time_ms=end_time
        )
        self.clips.append(timed_clip)
        
        if action not in ["INTERRUPT", "OVERLAP-START"]:
            self.current_time_ms = end_time
        else:
            self.current_time_ms = max(self.current_time_ms, end_time)
        
        return start_time
    
    def get_total_duration_ms(self) -> int:
        """Get total duration of the timeline"""
        if not self.clips:
            return 0
        return max(clip.end_time_ms for clip in self.clips)


class AudioProducer:
    """Main audio producer with noise reduction"""
    
    def __init__(self, openai_api_key: str, model: str = "tts-1-hd"):
        """
        Initialize Audio Producer with OpenAI TTS
        
        Args:
            openai_api_key: OpenAI API key
            model: TTS model ("tts-1" or "tts-1-hd" recommended)
        """
        self._check_ffmpeg()
        
        self.tts = TTSEngine(openai_api_key, model=model)
        self.composer = AudioComposer()
        self.cache_dir = Path("audio_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        print("Audio Producer initialized with noise reduction")
    
    def _check_ffmpeg(self):
        """Check if FFmpeg is installed"""
        import platform
        
        if shutil.which("ffmpeg") is None:
            print("\n" + "="*60)
            print("WARNING: FFmpeg not found!")
            print("="*60)
            system = platform.system()
            if system == "Windows":
                print("Install with: choco install ffmpeg")
            elif system == "Darwin":
                print("Install with: brew install ffmpeg")
            else:
                print("Install with: sudo apt-get install ffmpeg")
            print("="*60)
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    
    def configure_voices(self, host_a_voice: str = 'nova', host_b_voice: str = 'onyx'):
        """Configure voices for both hosts"""
        self.tts.configure_voices(host_a_voice=host_a_voice, host_b_voice=host_b_voice)
    
    def configure_noise_reduction(self, enabled: bool = True, 
                                   high_pass_hz: int = 80,
                                   low_pass_hz: int = 12000):
        """
        Configure noise reduction settings
        
        Args:
            enabled: Enable/disable noise reduction
            high_pass_hz: High-pass cutoff (default: 80Hz removes rumble)
            low_pass_hz: Low-pass cutoff (default: 12kHz removes hiss)
        """
        self.tts.configure_noise_reduction(
            enabled=enabled,
            high_pass_hz=high_pass_hz,
            low_pass_hz=low_pass_hz
        )
    
    def produce(self, script_json_path: str, output_path: str = "podcast.mp3",
                use_cache: bool = True, clear_cache_after: bool = True) -> str:
        """
        Complete audio production pipeline
        
        Args:
            script_json_path: Path to script JSON
            output_path: Path for final audio
            use_cache: Cache TTS clips during generation
            clear_cache_after: Clear cache after successful generation
            
        Returns:
            Path to final audio file
        """
        self._clear_cache_after = clear_cache_after
        
        print("\nPhase 3: Audio Production (WITH NOISE REDUCTION)")
        print("=" * 60)
        
        # Load script
        print("\nLoading script...")
        try:
            with open(script_json_path, 'r', encoding='utf-8') as f:
                script_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load script JSON: {e}")
        
        lines = script_data['lines']
        print(f"Loaded {len(lines)} dialogue lines")
        
        # Generate TTS for each line
        print("\nGenerating clean speech with noise reduction...")
        audio_clips = []
        
        for i, line_data in enumerate(lines, 1):
            text = line_data['text']
            speaker = line_data['speaker']
            emotion = line_data.get('emotion')
            
            # Cache handling
            cache_file = None
            if use_cache:
                cache_key = hashlib.md5(
                    f"{speaker}_{text}_{emotion}_{self.tts.model}_v2".encode()
                ).hexdigest()
                cache_file = self.cache_dir / f"{cache_key}.mp3"
            
            # Try cache first
            audio = None
            if use_cache and cache_file and cache_file.exists():
                try:
                    audio = AudioSegment.from_mp3(str(cache_file))
                    if audio is None or len(audio) == 0:
                        cache_file.unlink(missing_ok=True)
                        audio = None
                    else:
                        print(f"    [{i}/{len(lines)}] Cached: {speaker} - {text[:35]}...")
                except Exception:
                    cache_file.unlink(missing_ok=True)
                    audio = None
            
            # Generate fresh if needed
            if audio is None:
                try:
                    audio = self.tts.generate_speech(
                        text=text,
                        speaker=speaker,
                        emotion=emotion
                    )
                    
                    if audio is None or len(audio) == 0:
                        print(f"    [{i}/{len(lines)}] Empty audio, skipping...")
                        continue
                    
                    # Cache it
                    if use_cache and cache_file:
                        try:
                            audio.export(str(cache_file), format="mp3")
                        except Exception as e:
                            print(f"    [{i}/{len(lines)}] Cache write failed: {e}")
                    
                    emotion_tag = f"[{emotion}] " if emotion else ""
                    print(f"    [{i}/{len(lines)}] {speaker} {emotion_tag}- {text[:35]}...")
                    
                except Exception as e:
                    print(f"    [{i}/{len(lines)}] Failed: {e}")
                    continue
            
            # Rate limiting
            time.sleep(0.2)
            
            clip = AudioClip(
                speaker=speaker,
                text=text,
                audio=audio,
                emotion=emotion,
                action=line_data.get('action')
            )
            audio_clips.append(clip)
        
        print(f"Generated {len(audio_clips)} clean audio clips")
        
        if len(audio_clips) == 0:
            raise RuntimeError("No audio clips generated!")
        
        # Create timeline
        print("\nCreating conversation timeline...")
        timeline = ConversationTimeline()
        
        for clip, line_data in zip(audio_clips, lines):
            timeline.add_clip(
                clip=clip,
                timing_before=line_data.get('timing', 0.3),
                action=line_data.get('action')
            )
        
        total_duration_ms = timeline.get_total_duration_ms()
        minutes = total_duration_ms // 60000
        seconds = (total_duration_ms % 60000) // 1000
        print(f"Timeline created: {minutes}m {seconds}s")
        
        # Compose final audio
        final_audio = self.composer.compose(timeline, master_compression=True)
        
        if final_audio is None or len(final_audio) == 0:
            raise RuntimeError("Failed to compose audio!")
        
        print(f"\nFinal audio ready: {len(final_audio)/1000:.1f}s")
        
        # Export
        print(f"\nExporting to {output_path}...")
        try:
            final_audio.export(output_path, format="mp3", bitrate="192k")
        except Exception as e:
            print(f"Export failed: {e}")
            final_audio.export(output_path, format="mp3")
        
        # Verify
        if not os.path.exists(output_path):
            raise RuntimeError(f"Audio file not created at {output_path}")
        
        file_size = os.path.getsize(output_path)
        if file_size < 10000:
            raise RuntimeError(f"Audio file too small ({file_size} bytes)")
        
        file_size_mb = file_size / (1024 * 1024)
        print(f"Podcast exported: {file_size_mb:.2f} MB")
        print("🎉 Clean podcast generation complete!")
        
        if self._clear_cache_after:
            self.clear_cache()
        
        return output_path
    
    def clear_cache(self):
        """Clear the TTS cache directory"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
        print("🗑️ TTS audio cache cleared")

"""
Main orchestration script for the complete three-phase pipeline
Supports OpenAI and Groq LLMs, OpenAI TTS, with automatic cache cleanup
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import config
import config

# Import our three agents
from researcher import Researcher, EnrichedContent
from scriptwriter import Scriptwriter, PodcastScript
from audio_producer import AudioProducer


class PodcastGenerator:
    """
    Complete podcast generation pipeline orchestrator
    
    This class manages the three-phase process:
    1. Research & Enrichment (researcher.py)
    2. Script Generation (scriptwriter.py)
    3. Audio Production (audio_producer.py)
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, groq_api_key: Optional[str] = None,
                 output_dir: str = "./data/outputs", llm_provider: str = "openai"):
        """
        Initialize the podcast generator
        
        Args:
            openai_api_key: OpenAI API key for LLM and TTS
            groq_api_key: Groq API key for LLM (if using Groq)
            output_dir: Directory to store all outputs
            llm_provider: LLM provider to use ("openai" or "groq")
        """
        self.llm_provider = llm_provider.lower()
        
        # Determine which API key to use for LLM
        if self.llm_provider == "groq":
            if not groq_api_key:
                raise ValueError("Groq API key required when using Groq provider")
            self.llm_key = groq_api_key
            self.llm_model_research = config.GROQ_MODEL
            self.llm_model_script = config.GROQ_MODEL
            print(f"Using Groq LLM: {self.llm_model_research}")
        else:
            if not openai_api_key:
                raise ValueError("OpenAI API key required when using OpenAI provider")
            self.llm_key = openai_api_key
            self.llm_model_research = config.OPENAI_MODEL_RESEARCH
            self.llm_model_script = config.OPENAI_MODEL_SCRIPT
            print(f"Using OpenAI LLM: {self.llm_model_research}")
        
        # OpenAI API key is always needed for TTS
        if not openai_api_key:
            raise ValueError("OpenAI API key required for TTS")
        self.openai_key = openai_api_key
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.enriched_dir = self.output_dir / "enriched_content"
        self.scripts_dir = self.output_dir / "scripts"
        self.audio_dir = self.output_dir / "audio"
        
        for dir_path in [self.enriched_dir, self.scripts_dir, self.audio_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize agents
        self.researcher = Researcher(
            self.llm_key,
            model=self.llm_model_research,
            provider=self.llm_provider
        )
        self.scriptwriter = Scriptwriter(
            self.llm_key,
            model=self.llm_model_script,
            provider=self.llm_provider
        )
        self.audio_producer = AudioProducer(
            self.openai_key,
            model=config.TTS_MODEL
        )
        
        print("Podcast Generator initialized")
        print(f"Output directory: {self.output_dir.absolute()}")
    
    def generate_podcast(
        self,
        lecture_file: str,
        additional_notes: str = "",
        podcast_title: Optional[str] = None,
        skip_research: bool = False,
        skip_script: bool = False,
        skip_audio: bool = False,
        enriched_content_path: Optional[str] = None,
        script_path: Optional[str] = None
    ) -> dict:
        """
        Generate a complete podcast from lecture materials
        
        Args:
            lecture_file: Path to lecture file (PDF, PPTX, or TXT)
            additional_notes: Optional additional notes to include
            podcast_title: Title for the podcast (auto-generated if None)
            skip_research: Skip Phase 1 if enriched content already exists
            skip_script: Skip Phase 2 if script already exists
            skip_audio: Skip Phase 3 (useful for script-only generation)
            enriched_content_path: Path to existing enriched content (if skip_research=True)
            script_path: Path to existing script JSON (if skip_script=True)
        
        Returns:
            Dictionary with paths to all generated files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not podcast_title:
            file_name = Path(lecture_file).stem
            podcast_title = f"Learning Podcast: {file_name}"
        
        print(f"GENERATING PODCAST: {podcast_title}")
        print(f"LLM Provider: {self.llm_provider.upper()}")
        
        results = {
            'title': podcast_title,
            'timestamp': timestamp,
            'lecture_file': lecture_file,
            'llm_provider': self.llm_provider
        }
        
        # ============================================================
        # PHASE 1: RESEARCH & ENRICHMENT
        # ============================================================
        
        if not skip_research:
            print("PHASE 1: RESEARCH & ENRICHMENT")
            
            enriched_content = self.researcher.process(lecture_file, additional_notes)
            
            # Save enriched content
            enriched_txt_path = self.enriched_dir / f"enriched_{timestamp}.txt"
            self.researcher.save_result(enriched_content, str(enriched_txt_path))
            
            results['enriched_content_path'] = str(enriched_txt_path)
            results['enriched_content'] = enriched_content.final_document
        
        else:
            print("\n  Skipping Phase 1 (using existing enriched content)")
            
            if not skip_script:
                if not enriched_content_path:
                    raise ValueError("enriched_content_path required when skip_research=True (unless also skipping script)")
                
                with open(enriched_content_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "## Enriched Content" in content:
                        results['enriched_content'] = content.split("## Enriched Content")[1]
                    else:
                        results['enriched_content'] = content
                
                results['enriched_content_path'] = enriched_content_path
            else:
                results['enriched_content'] = None
                results['enriched_content_path'] = None
        
        # ============================================================
        # PHASE 2: SCRIPT GENERATION
        # ============================================================
        
        if not skip_script:
            print("PHASE 2: SCRIPT GENERATION")
            
            script = self.scriptwriter.create_script(
                enriched_content=results['enriched_content'],
                title=podcast_title
            )
            
            script_txt_path = self.scripts_dir / f"script_{timestamp}.txt"
            script_json_path = self.scripts_dir / f"script_{timestamp}.json"
            
            self.scriptwriter.save_script(script, str(script_txt_path))
            self.scriptwriter.export_for_audio(script, str(script_json_path))
            
            results['script_txt_path'] = str(script_txt_path)
            results['script_json_path'] = str(script_json_path)
        
        else:
            print("\n  Skipping Phase 2 (using existing script)")
            
            if not script_path:
                raise ValueError("script_path required when skip_script=True")
            
            results['script_json_path'] = script_path
        
        # ============================================================
        # PHASE 3: AUDIO PRODUCTION
        # ============================================================
        
        if not skip_audio:
            print("\n" + "🎵" * 20)
            print("PHASE 3: AUDIO PRODUCTION")
            print("🎵" * 20)
            
            audio_output_path = self.audio_dir / f"podcast_{timestamp}.mp3"
            
            self.audio_producer.produce(
                script_json_path=results['script_json_path'],
                output_path=str(audio_output_path),
                use_cache=config.USE_TTS_CACHE
            )
            
            results['audio_path'] = str(audio_output_path)
        
        else:
            print("\nSkipping Phase 3 (audio production)")
        
        # ============================================================
        # FINAL SUMMARY
        # ============================================================
        
        print("\n" + "=" * 60)
        print("PODCAST GENERATION COMPLETE!")
        print("=" * 60)
        print(f"\ Results:")
        print(f"   Title: {results['title']}")
        print(f"   LLM: {self.llm_provider.upper()}")
        
        if 'enriched_content_path' in results:
            print(f"  Enriched Content: {results['enriched_content_path']}")
        
        if 'script_txt_path' in results:
            print(f"  Script (Text): {results['script_txt_path']}")
        
        if 'script_json_path' in results:
            print(f"  Script (JSON): {results['script_json_path']}")
        
        if 'audio_path' in results:
            print(f"  Audio: {results['audio_path']}")
        
        results_json_path = self.output_dir / f"results_{timestamp}.json"
        with open(results_json_path, 'w', encoding='utf-8') as f:
            summary = {k: v for k, v in results.items() if k != 'enriched_content'}
            json.dump(summary, f, indent=2)
        
        print(f"\nResults summary saved: {results_json_path}")
        
        return results
    
    def generate_from_existing_enriched(self, enriched_content_path: str, 
                                       podcast_title: Optional[str] = None) -> dict:
        """
        Generate podcast from existing enriched content (skip Phase 1)
        
        Args:
            enriched_content_path: Path to enriched content file
            podcast_title: Optional title for the podcast
        
        Returns:
            Dictionary with paths to generated files
        """
        return self.generate_podcast(
            lecture_file="",
            podcast_title=podcast_title,
            skip_research=True,
            enriched_content_path=enriched_content_path
        )
    
    def generate_from_existing_script(self, script_json_path: str) -> dict:
        """
        Generate audio from existing script (skip Phases 1 & 2)
        
        Args:
            script_json_path: Path to script JSON file
        
        Returns:
            Dictionary with path to generated audio
        """
        return self.generate_podcast(
            lecture_file="",
            skip_research=True,
            skip_script=True,
            script_path=script_json_path
        )
    
    def script_only_mode(self, lecture_file: str, additional_notes: str = "",
                        podcast_title: Optional[str] = None) -> dict:
        """
        Generate only script without audio (Phases 1 & 2 only)
        
        Args:
            lecture_file: Path to lecture file
            additional_notes: Optional additional notes
            podcast_title: Optional title for the podcast
        
        Returns:
            Dictionary with paths to enriched content and script
        """
        return self.generate_podcast(
            lecture_file=lecture_file,
            additional_notes=additional_notes,
            podcast_title=podcast_title,
            skip_audio=True
        )


def main():
    """Command-line interface for the podcast generator"""
    
    parser = argparse.ArgumentParser(
        description="Generate educational podcasts from lecture materials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate complete podcast from PDF (OpenAI)
  python main.py lecture.pdf --title "Quantum Physics Basics"
  
  # Generate using Groq LLM (faster, cheaper)
  python main.py lecture.pdf --llm-provider groq
  
  # Generate script only (no audio)
  python main.py lecture.pptx --script-only
  
  # Generate from existing enriched content
  python main.py --from-enriched enriched.txt --title "My Podcast"
  
  # Generate audio from existing script
  python main.py --from-script script.json
  
  # Add additional notes
  python main.py lecture.pdf --notes "Focus on practical applications"
        """
    )
    
    parser.add_argument('lecture_file', nargs='?', help='Path to lecture file (PDF, PPTX, TXT)')
    parser.add_argument('--notes', help='Additional notes to include')
    parser.add_argument('--title', help='Podcast title (auto-generated if not provided)')
    parser.add_argument('--llm-provider', choices=['openai', 'groq'], default='openai',
                       help='LLM provider to use (default: openai)')
    parser.add_argument('--script-only', action='store_true', 
                       help='Generate script only, skip audio production')
    parser.add_argument('--from-enriched', metavar='PATH',
                       help='Use existing enriched content file')
    parser.add_argument('--from-script', metavar='PATH',
                       help='Use existing script JSON file')
    parser.add_argument('--output-dir', default='./data/outputs',
                       help='Output directory (default: ./data/outputs)')
    parser.add_argument('--openai-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--groq-key', help='Groq API key (or set GROQ_API_KEY env var)')
    
    args = parser.parse_args()
    
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    groq_key = args.groq_key or os.getenv('GROQ_API_KEY')
    
    if args.llm_provider == 'groq':
        if not groq_key:
            print("Error: Groq API key required when using Groq provider")
            print("   Set GROQ_API_KEY environment variable or use --groq-key")
            sys.exit(1)
        if not openai_key and not args.script_only:
            print("Error: OpenAI API key required for TTS")
            print("   Set OPENAI_API_KEY environment variable or use --openai-key")
            sys.exit(1)
    else:
        if not openai_key:
            print("Error: OpenAI API key required")
            print("   Set OPENAI_API_KEY environment variable or use --openai-key")
            sys.exit(1)
    
    if not args.from_enriched and not args.from_script and not args.lecture_file:
        print("Error: lecture_file required")
        print("   Usage: python main.py <lecture_file.pdf>")
        print("   Or use: --from-enriched or --from-script")
        sys.exit(1)
    
    try:
        generator = PodcastGenerator(
            openai_api_key=openai_key,
            groq_api_key=groq_key,
            output_dir=args.output_dir,
            llm_provider=args.llm_provider
        )
        
        if args.from_script:
            results = generator.generate_from_existing_script(args.from_script)
        
        elif args.from_enriched:
            results = generator.generate_from_existing_enriched(
                enriched_content_path=args.from_enriched,
                podcast_title=args.title
            )
        
        elif args.script_only:
            results = generator.script_only_mode(
                lecture_file=args.lecture_file,
                additional_notes=args.notes or "",
                podcast_title=args.title
            )
        
        else:
            results = generator.generate_podcast(
                lecture_file=args.lecture_file,
                additional_notes=args.notes or "",
                podcast_title=args.title
            )
        
        print("\nSuccess! Check the output directory for results.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
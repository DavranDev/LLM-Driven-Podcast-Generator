import os
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ddgs import DDGS  
import fitz  # PyMuPDF
from pptx import Presentation


@dataclass
class ResearchGap:
    topic: str
    question: str
    context: str
    priority: int


@dataclass
class EnrichedContent:
    original_content: str
    gaps_found: List[ResearchGap]
    enriched_sections: List[Dict[str, str]]
    final_document: str
    metadata: Dict


class ContentExtractor:
    
    @staticmethod
    def extract_from_pdf(pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            full_text = []
            
            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text")
                
                full_text.append(f"\n--- Page {page_num} ---\n")
                full_text.append(text)
                
                images = page.get_images()
                if images:
                    full_text.append(f"\n[Page contains {len(images)} image(s)]\n")
            
            doc.close()
            return "\n".join(full_text)
        
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""
    
    @staticmethod
    def extract_from_pptx(pptx_path: str) -> str:
        """Extract text from PowerPoint with slide structure"""
        try:
            prs = Presentation(pptx_path)
            full_text = []
            
            for slide_num, slide in enumerate(prs.slides, 1):
                full_text.append(f"\n--- Slide {slide_num} ---\n")
                
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        full_text.append(shape.text)
                    
                    if shape.has_table:
                        table = shape.table
                        for row in table.rows:
                            row_text = " | ".join([cell.text for cell in row.cells])
                            full_text.append(row_text)
                
                if slide.has_notes_slide:
                    notes_text = slide.notes_slide.notes_text_frame.text
                    if notes_text:
                        full_text.append(f"\n[Speaker Notes: {notes_text}]\n")
            
            return "\n".join(full_text)
        
        except Exception as e:
            print(f"Error extracting PPTX: {e}")
            return ""
    
    @staticmethod
    def extract_from_text(txt_path: str) -> str:
        """Extract text from plain text files"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""


class GapAnalyzer:
    """Analyzes content for knowledge gaps and ambiguities"""
    
    def __init__(self, llm):
        self.llm = llm
        self.gap_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert educational content analyzer. Your job is to identify knowledge gaps, 
            ambiguities, and undefined terms in lecture materials.
            
            For each gap you find, identify:
            1. The topic or term that needs clarification
            2. A specific question that would fill the gap
            3. The surrounding context
            4. Priority level (1-5, where 5 is critical to understanding)
            
            Focus on:
            - Unexplained acronyms or technical terms
            - Historical references without context (dates, events, protocols)
            - Concepts mentioned but not defined
            - Statistical claims without sources
            - Complex ideas that lack examples or analogies
            
            Return your analysis as a JSON array of gaps."""),
            ("user", "Analyze this lecture content for knowledge gaps:\n\n{content}")
        ])
    
    def analyze(self, content: str, max_gaps: int = 10) -> List[ResearchGap]:
        """Identify knowledge gaps in the content"""
        try:
            if len(content) > 12000:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=10000,
                    chunk_overlap=500
                )
                chunks = splitter.split_text(content)
                
                all_gaps = []
                for chunk in chunks[:3]:
                    gaps = self._analyze_chunk(chunk)
                    all_gaps.extend(gaps)
                
                return self._deduplicate_gaps(all_gaps)[:max_gaps]
            else:
                return self._analyze_chunk(content)[:max_gaps]
        
        except Exception as e:
            print(f"Error in gap analysis: {e}")
            return []
    
    def _analyze_chunk(self, content: str) -> List[ResearchGap]:
        """Analyze a single chunk of content"""
        try:
            chain = self.gap_analysis_prompt | self.llm
            response = chain.invoke({"content": content})
            
            content_text = response.content
            
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = content_text
            
            gaps_data = json.loads(json_str)
            
            gaps = []
            for gap_data in gaps_data:
                gap = ResearchGap(
                    topic=gap_data.get('topic', ''),
                    question=gap_data.get('question', ''),
                    context=gap_data.get('context', ''),
                    priority=gap_data.get('priority', 3)
                )
                gaps.append(gap)
            
            return sorted(gaps, key=lambda x: x.priority, reverse=True)
        
        except Exception as e:
            print(f"Error analyzing chunk: {e}")
            return []
    
    def _deduplicate_gaps(self, gaps: List[ResearchGap]) -> List[ResearchGap]:
        """Remove duplicate or very similar gaps"""
        unique_gaps = []
        seen_topics = set()
        
        for gap in sorted(gaps, key=lambda x: x.priority, reverse=True):
            topic_lower = gap.topic.lower()
            if topic_lower not in seen_topics:
                unique_gaps.append(gap)
                seen_topics.add(topic_lower)
        
        return unique_gaps


class WebSearcher:
    """Searches the web to fill knowledge gaps"""
    
    def __init__(self, max_results: int = 3, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """Perform a web search and return results"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
            
            return [
                {
                    'title': r.get('title', ''),
                    'body': r.get('body', ''),
                    'url': r.get('href', r.get('url', ''))
                }
                for r in results
            ]
        except Exception as e:
            print(f"Search error for '{query}': {e}")
            return []
    
    def fill_gap(self, gap: ResearchGap) -> Optional[Dict[str, str]]:
        """Search for information to fill a knowledge gap"""
        search_query = f"{gap.topic} {gap.question}"
        
        results = self.search(search_query)
        
        if not results:
            return None
        
        combined_content = "\n".join([
            f"Source: {r['title']}\n{r['body']}"
            for r in results
        ])
        
        return {
            'gap': gap.topic,
            'question': gap.question,
            'summary': combined_content[:1500],
            'sources': [r['url'] for r in results if r['url']]
        }


class ContentEnricher:
    """Enriches content with research findings"""
    
    def __init__(self, llm):
        self.llm = llm
        self.enrichment_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert content enricher. Your job is to integrate additional research 
            into lecture materials seamlessly.
            
            Given:
            1. Original lecture content
            2. Knowledge gaps identified
            3. Research findings for each gap
            
            Create a comprehensive, enriched document that:
            - Maintains the original structure and flow
            - Integrates research findings naturally at appropriate points
            - Adds context and explanations for technical terms
            - Includes relevant examples and analogies
            - Maintains an educational, engaging tone
            
            The output should read as if it was originally written this way, not like patches were added."""),
            ("user", """Original Content:
{original_content}

Knowledge Gaps & Research Findings:
{enrichments}

Create an enriched version of this content that seamlessly integrates the research findings.""")
        ])
    
    def enrich(self, original_content: str, enrichments: List[Dict[str, str]]) -> str:
        """Create an enriched version of the content"""
        try:
            enrichments_text = self._format_enrichments(enrichments)
            
            if len(original_content) > 10000:
                return self._enrich_in_sections(original_content, enrichments)
            
            chain = self.enrichment_prompt | self.llm
            response = chain.invoke({
                "original_content": original_content,
                "enrichments": enrichments_text
            })
            
            return response.content
        
        except Exception as e:
            print(f"Error enriching content: {e}")
            return original_content
    
    def _format_enrichments(self, enrichments: List[Dict[str, str]]) -> str:
        """Format enrichments for the prompt"""
        formatted = []
        for i, enr in enumerate(enrichments, 1):
            formatted.append(f"\n{i}. Topic: {enr['gap']}")
            formatted.append(f"   Question: {enr['question']}")
            formatted.append(f"   Research: {enr['summary']}\n")
        
        return "\n".join(formatted)
    
    def _enrich_in_sections(self, content: str, enrichments: List[Dict[str, str]]) -> str:
        """Enrich content in sections for long documents"""
        sections = re.split(r'--- Page \d+ ---', content)
        enriched_sections = []
        
        enrichments_per_section = len(enrichments) // len(sections) + 1
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            start_idx = i * enrichments_per_section
            end_idx = start_idx + enrichments_per_section
            section_enrichments = enrichments[start_idx:end_idx]
            
            if section_enrichments:
                enriched = self.enrich(section, section_enrichments)
                enriched_sections.append(enriched)
            else:
                enriched_sections.append(section)
        
        return "\n\n".join(enriched_sections)


class Researcher:
    """Main researcher agent orchestrating the entire Phase 1"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", provider: str = "openai"):
        """
        Initialize the Researcher
        
        Args:
            api_key: API key for the LLM provider
            model: Model name (e.g., "gpt-4o-mini" for OpenAI or "llama-3.3-70b-versatile" for Groq)
            provider: LLM provider ("openai" or "groq")
        """
        self.provider = provider.lower()
        
        if self.provider == "groq":
            self.llm = ChatGroq(
                api_key=api_key,
                model=model,
                temperature=0.3
            )
            print(f"Researcher using Groq LLM: {model}")
        else:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0.3
            )
            print(f"Researcher using OpenAI LLM: {model}")
        
        self.extractor = ContentExtractor()
        self.gap_analyzer = GapAnalyzer(self.llm)
        self.web_searcher = WebSearcher()
        self.enricher = ContentEnricher(self.llm)
    
    def process(self, file_path: str, additional_notes: str = "") -> EnrichedContent:
        """
        Complete research pipeline:
        1. Extract content
        2. Analyze for gaps
        3. Search for missing information
        4. Enrich the content
        """
        print("Phase 1: Research & Enrichment")
        print("=" * 50)
        
        print("\nExtracting content...")
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            content = self.extractor.extract_from_pdf(file_path)
        elif file_ext in ['.pptx', '.ppt']:
            content = self.extractor.extract_from_pptx(file_path)
        elif file_ext == '.txt':
            content = self.extractor.extract_from_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        if additional_notes:
            content += f"\n\n--- Additional Notes ---\n{additional_notes}"
        
        print(f"Extracted {len(content)} characters")
        
        print("\n🔎 Analyzing for knowledge gaps...")
        gaps = self.gap_analyzer.analyze(content)
        print(f"Found {len(gaps)} knowledge gaps")
        
        for i, gap in enumerate(gaps, 1):
            print(f"   {i}. {gap.topic} (Priority: {gap.priority}/5)")
        
        print("\nSearching for missing information...")
        enrichments = []
        
        for gap in gaps:
            print(f"   Searching: {gap.topic}...")
            enrichment = self.web_searcher.fill_gap(gap)
            
            if enrichment:
                enrichments.append(enrichment)
                print(f"Found {len(enrichment['sources'])} sources")
            else:
                print(f"No results found")
        
        print("\nCreating enriched document...")
        final_document = self.enricher.enrich(content, enrichments)
        print(f"Enriched document created ({len(final_document)} characters)")
        
        result = EnrichedContent(
            original_content=content,
            gaps_found=gaps,
            enriched_sections=enrichments,
            final_document=final_document,
            metadata={
                'file_path': file_path,
                'original_length': len(content),
                'enriched_length': len(final_document),
                'gaps_found': len(gaps),
                'gaps_filled': len(enrichments),
                'llm_provider': self.provider
            }
        )
        
        print("\nPhase 1 Complete!")
        print("=" * 50)
        
        return result
    
    def save_result(self, result: EnrichedContent, output_path: str):
        """Save the enriched content to a file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# ENRICHED LECTURE CONTENT\n\n")
            f.write(f"## Metadata\n")
            f.write(f"- Original Length: {result.metadata['original_length']} chars\n")
            f.write(f"- Enriched Length: {result.metadata['enriched_length']} chars\n")
            f.write(f"- Gaps Found: {result.metadata['gaps_found']}\n")
            f.write(f"- Gaps Filled: {result.metadata['gaps_filled']}\n")
            f.write(f"- LLM Provider: {result.metadata.get('llm_provider', 'unknown')}\n\n")
            
            f.write("## Knowledge Gaps Identified\n\n")
            for i, gap in enumerate(result.gaps_found, 1):
                f.write(f"{i}. **{gap.topic}** (Priority: {gap.priority}/5)\n")
                f.write(f"   - Question: {gap.question}\n\n")
            
            f.write("\n## Enriched Content\n\n")
            f.write(result.final_document)
        
        print(f"Saved enriched content to: {output_path}")
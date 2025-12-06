"""
Scriptwriter Agent - Phase 2
Generates a natural, conversational podcast script with realistic speech patterns.
Supports both OpenAI and Groq LLMs.

Key improvements:
- Realistic filler words (um, uh, like, you know)
- Backchannel responses (mhm, right, yeah, exactly)
- Concise responses (not too long)
- Natural interruptions and overlaps
- Real podcast conversation dynamics
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate


@dataclass
class DialogueLine:
    """Represents a single line of dialogue"""
    speaker: str  # "HOST_A" or "HOST_B"
    text: str
    emotion: Optional[str] = None  # LAUGH, SIGH, EXCITED, THOUGHTFUL, etc.
    action: Optional[str] = None  # INTERRUPT, PAUSE, OVERLAP
    timing: Optional[float] = None  # Delay in seconds before this line


@dataclass
class PodcastScript:
    """The complete podcast script"""
    title: str
    duration_estimate: int  # in seconds
    intro: List[DialogueLine]
    main_content: List[DialogueLine]
    outro: List[DialogueLine]
    metadata: Dict
    
    def get_all_lines(self) -> List[DialogueLine]:
        """Get all dialogue lines in order"""
        return self.intro + self.main_content + self.outro


class PersonaDesigner:
    """Designs the personas for the two podcast hosts"""
    
    @staticmethod
    def get_host_a_persona() -> Dict[str, str]:
        """
        Host A: The Curious Student (The Proxy)
        Represents the listener - asks questions, seeks clarifications
        """
        return {
            "name": "Alex",
            "role": "The Curious Student",
            "traits": [
                "Genuinely curious and eager to understand",
                "Makes connections to real-world examples",
                "Not afraid to admit confusion",
                "Uses analogies to check understanding",
                "Reacts naturally with 'oh wow', 'that's wild', 'wait really?'",
                "Uses natural fillers: 'um', 'like', 'I mean', 'you know'",
                "Gives backchannel responses when listening: 'mhm', 'right', 'okay'"
            ],
            "speech_patterns": [
                "Starts with 'So wait...' or 'Okay so...'",
                "Says 'that's kinda like...' for analogies",
                "Interrupts with 'Oh! So it's basically...'",
                "Uses 'I mean' to self-correct mid-sentence",
                "Trails off with 'so that means...' inviting explanation",
                "Says 'huh' or 'oh interesting' as reactions"
            ],
            "backchannel_phrases": [
                "mhm",
                "right",
                "yeah",
                "okay okay",
                "oh wow",
                "interesting",
                "got it"
            ]
        }
    
    @staticmethod
    def get_host_b_persona() -> Dict[str, str]:
        """
        Host B: The Knowledgeable Guide (The Expert)
        Represents the teacher - explains clearly, validates understanding
        """
        return {
            "name": "Jordan",
            "role": "The Knowledgeable Guide",
            "traits": [
                "Patient and encouraging but conversational",
                "Validates good questions naturally: 'Yeah exactly' or 'Right so...'",
                "Gently corrects: 'Well, kinda... it's more like...'",
                "Uses natural fillers when thinking: 'um', 'so', 'like'",
                "Gives brief responses, not lectures",
                "Builds on partner's points: 'Yeah and the cool thing is...'",
                "Sometimes pauses to think: 'hmm let me think...'"
            ],
            "speech_patterns": [
                "Says 'So basically...' to simplify",
                "Uses 'Right, so...' to build on points",
                "Validates with 'Exactly' or 'Yeah that's it'",
                "Corrects with 'Well... not quite' or 'So actually...'",
                "Explains with 'Think of it like...'",
                "Trails off inviting response: '...you know what I mean?'"
            ],
            "backchannel_phrases": [
                "exactly",
                "right right",
                "yeah",
                "mm",
                "sure sure"
            ]
        }


# =============================================================================
# REALISTIC DIALOGUE PROMPT TEMPLATES
# =============================================================================

REALISTIC_DIALOGUE_SYSTEM_PROMPT = """You are an expert podcast dialogue writer who creates ULTRA-REALISTIC conversations.
Your dialogue should sound like two real friends talking, NOT like a scripted show.

HOST PERSONAS:
{host_a_name} ({host_a_role}):
{host_a_traits}
Speech patterns: {host_a_patterns}

{host_b_name} ({host_b_role}):
{host_b_traits}  
Speech patterns: {host_b_patterns}

=== CRITICAL REALISM RULES ===

1. FILLER WORDS (use naturally, not excessively):
   - "um", "uh" when thinking
   - "like" as a verbal pause
   - "I mean" for self-correction
   - "you know" to check understanding
   - "so" to start explanations
   
2. BACKCHANNEL RESPONSES (the listener gives feedback):
   - While one person explains, the other says: "mhm", "right", "yeah", "okay"
   - These should be SHORT separate lines, not part of longer dialogue
   - Example:
     HOST_B: So the thing is, neural networks basically learn patterns...
     HOST_A: mhm
     HOST_B: ...and they adjust these weights over time.
     HOST_A: oh okay okay

3. KEEP RESPONSES SHORT (this is crucial!):
   - NO response should be more than 2-3 sentences
   - Long explanations should be broken up by the other host reacting
   - After 2 sentences, the other person should react or ask something
   
4. NATURAL INTERRUPTIONS:
   - "Oh! So it's like..." 
   - "Wait wait wait..."
   - "Hold on, so..."
   - Use [INTERRUPT] marker for these

5. REALISTIC REACTIONS:
   - "That's wild"
   - "Huh, I never thought of it that way"
   - "Oh that makes so much more sense"
   - "Wait, really?"
   - [LAUGH] when something is amusing or relatable

6. INCOMPLETE SENTENCES:
   - People trail off: "so that means..."
   - People get cut off mid-sentence
   - People restart: "I mean, it's like... no wait, think of it this way"

7. AVOID THESE (they sound scripted):
   - "Great question!" (too formal)
   - "That's a fascinating point" (too formal) 
   - "Let me explain" (too lecture-y)
   - "As you mentioned" (too formal)
   - Long uninterrupted monologues

=== MARKERS TO USE ===
Emotions: [LAUGH], [CHUCKLE], [EXCITED], [THOUGHTFUL]
Actions: [INTERRUPT], [OVERLAP-START], [OVERLAP-END], [PAUSE], [BEAT]

=== FORMAT ===
HOST_A: [markers if any] Dialogue text
HOST_B: [markers if any] Dialogue text

=== EXAMPLE OF GOOD REALISTIC DIALOGUE ===

HOST_A: So wait, you're telling me that like... the AI just figures this out on its own?
HOST_B: Yeah, I mean... sort of. It's more like it sees patterns, you know?
HOST_A: mhm
HOST_B: And then it just... adjusts based on what works.
HOST_A: [BEAT] Huh. That's kinda like how we learn, right? Like trial and error?
HOST_B: Exactly! Yeah that's actually a really good way to think about it.
HOST_A: [LAUGH] Okay okay, I think I'm getting it now.
HOST_B: So the cool thing is...
HOST_A: [INTERRUPT] Oh wait, so is that why it sometimes gets stuff wrong?
HOST_B: [CHUCKLE] Yeah, um, basically. It's still learning, you know?

=== BAD EXAMPLE (too scripted) ===
HOST_A: Can you explain how artificial intelligence learns?
HOST_B: Certainly! That's a great question. Artificial intelligence learns through a process called machine learning. Essentially, the AI is exposed to large amounts of data and it identifies patterns within that data. Over time, it adjusts its internal parameters to better recognize these patterns.

The bad example is too formal, too long, no reactions, no fillers, sounds like a lecture.

Now write the dialogue for the segment below."""


class ConversationArchitect:
    """Designs the structure and flow of the conversation"""
    
    def __init__(self, llm):
        self.llm = llm
        self.outline_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a podcast conversation architect. Given lecture content, create a compelling 
            outline for a conversational podcast that feels like two friends chatting.
            
            Your outline should:
            1. Start with a hook - something surprising or intriguing
            2. Break content into 5-7 SHORT digestible segments
            3. Each segment should be 3-5 exchanges maximum
            4. Include moments for reactions and "aha" moments
            5. Mark places where the listener would naturally have questions
            
            Return a structured outline with:
            - Intro hook (1 punchy sentence)
            - Main segments (each with 2-3 key points only)
            - Outro summary (1-2 sentences)
            
            Keep it conversational. Think "coffee shop chat" not "lecture hall"."""),
            ("user", "Create a podcast outline for this content:\n\n{content}")
        ])
    
    def create_outline(self, content: str) -> Dict:
        """Create a structured outline for the podcast"""
        try:
            chain = self.outline_prompt | self.llm
            response = chain.invoke({"content": content})
            
            # Parse the outline
            outline_text = response.content
            
            # Try to extract JSON if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', outline_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                return self._parse_text_outline(outline_text)
        
        except Exception as e:
            print(f"Error creating outline: {e}")
            return self._default_outline(content)
    
    def _parse_text_outline(self, text: str) -> Dict:
        """Parse a text-based outline into structured format"""
        sections = {
            'intro_hook': '',
            'segments': [],
            'outro': ''
        }
        
        current_section = None
        current_segment = None
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if 'intro' in line.lower() or 'hook' in line.lower():
                current_section = 'intro_hook'
            elif 'outro' in line.lower() or 'conclusion' in line.lower():
                current_section = 'outro'
            elif 'segment' in line.lower() or re.match(r'^\d+\.', line):
                if current_segment:
                    sections['segments'].append(current_segment)
                current_segment = {'title': line, 'points': []}
                current_section = 'segment'
            elif current_section == 'intro_hook':
                sections['intro_hook'] += ' ' + line
            elif current_section == 'outro':
                sections['outro'] += ' ' + line
            elif current_section == 'segment' and current_segment:
                current_segment['points'].append(line)
        
        if current_segment:
            sections['segments'].append(current_segment)
        
        return sections
    
    def _default_outline(self, content: str) -> Dict:
        """Create a basic outline if automated parsing fails"""
        return {
            'intro_hook': 'Welcome to today\'s discussion!',
            'segments': [
                {'title': 'Core Concepts', 'points': ['Main ideas']},
                {'title': 'Deep Dive', 'points': ['Detailed exploration']},
                {'title': 'Real-World Applications', 'points': ['Practical examples']}
            ],
            'outro': 'That wraps up our discussion!'
        }


class DialogueGenerator:
    """Generates natural, conversational dialogue"""
    
    def __init__(self, llm):
        self.llm = llm
        self.host_a = PersonaDesigner.get_host_a_persona()
        self.host_b = PersonaDesigner.get_host_b_persona()
        
        self.dialogue_prompt = ChatPromptTemplate.from_messages([
            ("system", REALISTIC_DIALOGUE_SYSTEM_PROMPT),
            ("user", """Segment: {segment_title}

Key Points to Cover (briefly, conversationally):
{segment_points}

Source Material (for reference):
{source_content}

Write a natural, realistic conversation covering these points. Remember:
- Keep each response SHORT (2-3 sentences max)
- Include backchannel responses (mhm, right, yeah)
- Use fillers naturally (um, like, so)
- Let them interrupt and react to each other
- This should feel like friends chatting, not a lecture""")
        ])
    
    def generate_intro(self, hook: str, title: str = "") -> List[DialogueLine]:
        """Generate the introduction dialogue with greeting, topic intro, and motivation"""
        intro_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Write a podcast introduction (8-12 lines of dialogue) that feels like two friends starting a conversation.

STRUCTURE (follow this order):
1. GREETING (2-3 lines): {self.host_a['name']} welcomes listeners casually, {self.host_b['name']} responds warmly
2. TOPIC TEASE (2-3 lines): Introduce what we're talking about today with genuine excitement
3. REAL-WORLD HOOK (3-4 lines): Give a relatable real-world example or "why should I care" moment
4. TRANSITION (1-2 lines): Set up the deep dive

HOSTS:
- {self.host_a['name']} (curious student): Welcomes listeners, gets excited about the topic, asks "wait, so like..."
- {self.host_b['name']} (knowledgeable guide): Builds hype, gives the motivating example, teases what's coming

REAL-WORLD EXAMPLES TO USE (pick one that fits):
- "You know how Netflix always knows what you wanna watch?"
- "Ever wonder how Spotify creates those crazy accurate playlists?"
- "You know when Amazon suggests something and you're like 'how did they know?'"
- "Think about how TikTok's For You page is scary accurate..."
- Reference any technology people use daily

TOPIC/HOOK: {hook}

REALISM RULES:
- Use fillers: "um", "like", "so", "you know"
- Include reactions: "oh wow", "that's wild", "wait really?"
- Keep each line SHORT (1-2 sentences max)
- Sound like excited friends, NOT news anchors
- {self.host_a['name']} should react with genuine curiosity
- Include at least one [LAUGH] or [EXCITED] marker

FORMAT:
HOST_A: [markers if any] Dialogue text
HOST_B: [markers if any] Dialogue text

EXAMPLE OF GOOD INTRO:
HOST_A: Hey everyone! Welcome back, so glad you're here with us today.
HOST_B: Yeah, we've got a really cool one for you today.
HOST_A: Okay so, {self.host_b['name']}, I've been dying to talk about this... like, you know how Netflix always seems to know exactly what I wanna watch?
HOST_B: [LAUGH] Yeah, it's kinda creepy sometimes, right?
HOST_A: Totally! So today we're diving into how that actually works.
HOST_B: [EXCITED] Yeah, and trust me, it's way cooler than you'd think. We're talking about recommendation systems and specifically this thing called...
HOST_A: ooh okay, I'm hooked already."""),
            ("user", f"Write the intro dialogue for a podcast about: {hook}\nPodcast title: {title}\n\nRemember: greeting first, then topic intro with a real-world example that hooks the listener!")
        ])
        
        try:
            chain = intro_prompt | self.llm
            response = chain.invoke({"hook": hook, "title": title})
            return self._parse_dialogue(response.content)
        except Exception as e:
            print(f"Error generating intro: {e}")
            return self._default_intro()
    
    def generate_outro(self, title: str = "") -> List[DialogueLine]:
        """Generate the conclusion dialogue with summary and sign-off"""
        outro_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Write a podcast conclusion (6-10 lines of dialogue) that wraps up naturally.

STRUCTURE (follow this order):
1. SIGNAL WRAP-UP (1-2 lines): "Alright, so..." or "Okay, so to wrap this up..."
2. KEY TAKEAWAYS (2-3 lines): {self.host_a['name']} summarizes 1-2 things they learned (genuinely excited)
3. FINAL THOUGHT (2-3 lines): {self.host_b['name']} adds one "big picture" insight or future implication
4. SIGN-OFF (2-3 lines): Thank listeners, tease next episode or encourage engagement

HOSTS:
- {self.host_a['name']} (curious student): Summarizes what blew their mind, sounds genuinely excited about what they learned
- {self.host_b['name']} (knowledgeable guide): Validates, adds final wisdom, thanks listeners warmly

REALISM RULES:
- Use fillers: "um", "like", "so", "you know"  
- Keep each line SHORT (1-2 sentences max)
- Sound like friends wrapping up a great conversation
- Include genuine enthusiasm
- {self.host_a['name']} should sound like they genuinely learned something cool

FORMAT:
HOST_A: [markers if any] Dialogue text
HOST_B: [markers if any] Dialogue text

EXAMPLE OF GOOD OUTRO:
HOST_A: Okay so like, my mind is kinda blown right now.
HOST_B: [LAUGH] Yeah, it's pretty cool stuff, right?
HOST_A: So basically, the big takeaway is that these systems are learning from us all the time, and they're getting scary good at it.
HOST_B: Exactly. And I think the exciting part is, this is just the beginning, you know?
HOST_A: [EXCITED] Yeah! Like, imagine where this'll be in five years.
HOST_B: Right. Anyway, thanks so much for hanging out with us today, everyone.
HOST_A: Yeah, seriously, you guys are the best. Let us know what you thought!
HOST_B: See you next time!"""),
            ("user", f"Write the outro dialogue for the podcast: {title}\n\nRemember: summarize key insights, sound genuinely excited, and thank the listeners!")
        ])
        
        try:
            chain = outro_prompt | self.llm
            response = chain.invoke({"title": title})
            return self._parse_dialogue(response.content)
        except Exception as e:
            print(f"Error generating outro: {e}")
            return self._default_outro()
    
    def _default_intro(self) -> List[DialogueLine]:
        """Fallback intro if generation fails"""
        return [
            DialogueLine(speaker="HOST_A", text="Hey everyone! Welcome back to the show.", timing=0.2),
            DialogueLine(speaker="HOST_B", text="Yeah, we've got a really interesting topic for you today.", timing=0.2),
            DialogueLine(speaker="HOST_A", text="I'm super excited about this one, let's dive in!", timing=0.2),
        ]
    
    def _default_outro(self) -> List[DialogueLine]:
        """Fallback outro if generation fails"""
        return [
            DialogueLine(speaker="HOST_A", text="Alright, that was awesome. I learned so much.", timing=0.2),
            DialogueLine(speaker="HOST_B", text="Yeah, thanks so much for listening everyone!", timing=0.2),
            DialogueLine(speaker="HOST_A", text="See you next time!", timing=0.2),
        ]
    
    def generate_segment(self, segment_title: str, segment_points: List[str],
                        source_content: str) -> List[DialogueLine]:
        """Generate dialogue for a content segment"""
        try:
            chain = self.dialogue_prompt | self.llm
            response = chain.invoke({
                "host_a_name": self.host_a['name'],
                "host_a_role": self.host_a['role'],
                "host_a_traits": "\n".join(f"- {t}" for t in self.host_a['traits']),
                "host_a_patterns": ", ".join(self.host_a['speech_patterns']),
                "host_b_name": self.host_b['name'],
                "host_b_role": self.host_b['role'],
                "host_b_traits": "\n".join(f"- {t}" for t in self.host_b['traits']),
                "host_b_patterns": ", ".join(self.host_b['speech_patterns']),
                "segment_title": segment_title,
                "segment_points": "\n".join(f"- {p}" for p in segment_points),
                "source_content": source_content[:2000]  # Limit context length
            })
            
            dialogue_lines = self._parse_dialogue(response.content)
            return dialogue_lines
        
        except Exception as e:
            print(f"Error generating segment: {e}")
            return []
    
    def _parse_dialogue(self, dialogue_text: str) -> List[DialogueLine]:
        """Parse raw dialogue text into structured DialogueLine objects"""
        lines = []
        
        # Split by speaker lines
        pattern = r'(HOST_[AB]):\s*(.*?)(?=HOST_[AB]:|$)'
        matches = re.finditer(pattern, dialogue_text, re.DOTALL)
        
        for match in matches:
            speaker = match.group(1)
            text = match.group(2).strip()
            
            if not text:
                continue
            
            # Extract emotion/action tags
            emotion = None
            action = None
            
            # Check for emotion tags
            emotion_match = re.search(r'\[(LAUGH|CHUCKLE|SIGH|EXCITED|THOUGHTFUL|AMAZED)\]', text)
            if emotion_match:
                emotion = emotion_match.group(1)
                text = text.replace(f'[{emotion}]', '').strip()
            
            # Check for action tags
            action_match = re.search(r'\[(INTERRUPT|OVERLAP-START|OVERLAP-END|PAUSE|BEAT)\]', text)
            if action_match:
                action = action_match.group(1)
                text = text.replace(f'[{action}]', '').strip()
            
            # Default timing between lines - shorter for natural flow
            timing = 0.2 if not action else (1.0 if action == 'PAUSE' else 0.08)
            
            # Even shorter timing for backchannel responses
            if text.lower() in ['mhm', 'right', 'yeah', 'okay', 'mm', 'uh huh', 'got it', 'interesting']:
                timing = 0.05
            
            line = DialogueLine(
                speaker=speaker,
                text=text,
                emotion=emotion,
                action=action,
                timing=timing
            )
            lines.append(line)
        
        return lines


class Scriptwriter:
    """Main scriptwriter agent orchestrating Phase 2"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", provider: str = "openai"):
        """
        Initialize the Scriptwriter
        
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
                temperature=0.85  # Higher temperature for creative dialogue
            )
            print(f"Scriptwriter using Groq LLM: {model}")
        else:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=0.85  # Higher temperature for creative dialogue
            )
            print(f"Scriptwriter using OpenAI LLM: {model}")
        
        self.architect = ConversationArchitect(self.llm)
        self.generator = DialogueGenerator(self.llm)
    
    def create_script(self, enriched_content: str, title: str = "Learning Podcast") -> PodcastScript:
        """
        Complete scriptwriting pipeline:
        1. Create conversation outline
        2. Generate intro
        3. Generate segments
        4. Generate outro
        """
        print("\nPhase 2: Script Generation")
        print("=" * 50)
        
        # Step 1: Create outline
        print("\nCreating conversation outline...")
        outline = self.architect.create_outline(enriched_content)
        print(f"Outline created with {len(outline.get('segments', []))} segments")
        
        # Step 2: Generate intro
        print("\nGenerating introduction...")
        intro_lines = self.generator.generate_intro(outline.get('intro_hook', ''), title=title)
        print(f"Intro generated ({len(intro_lines)} lines)")
        
        # Step 3: Generate main content segments
        print("\nGenerating dialogue for segments...")
        all_segment_lines = []
        
        for i, segment in enumerate(outline.get('segments', []), 1):
            print(f"   Segment {i}: {segment.get('title', 'Untitled')}")
            
            segment_lines = self.generator.generate_segment(
                segment_title=segment.get('title', ''),
                segment_points=segment.get('points', []),
                source_content=enriched_content
            )
            
            all_segment_lines.extend(segment_lines)
            print(f"Generated {len(segment_lines)} lines")
        
        outro_lines = self.generator.generate_outro(title=title)
        print(f"Outro generated ({len(outro_lines)} lines)")
        
        total_lines = len(intro_lines) + len(all_segment_lines) + len(outro_lines)
        avg_words_per_line = 12  
        words_per_minute = 160  
        duration_estimate = int((total_lines * avg_words_per_line) / words_per_minute * 60)
        
        # Create the script
        script = PodcastScript(
            title=title,
            duration_estimate=duration_estimate,
            intro=intro_lines,
            main_content=all_segment_lines,
            outro=outro_lines,
            metadata={
                'segments': len(outline.get('segments', [])),
                'total_lines': total_lines,
                'intro_lines': len(intro_lines),
                'content_lines': len(all_segment_lines),
                'outro_lines': len(outro_lines),
                'llm_provider': self.provider
            }
        )
        
        print(f"\nScript complete!")
        print(f"   Total lines: {total_lines}")
        print(f"   Estimated duration: {duration_estimate // 60}m {duration_estimate % 60}s")
        print("=" * 50)
        
        return script
    
    def save_script(self, script: PodcastScript, output_path: str):
        """Save the script to a file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {script.title}\n\n")
            f.write(f"**Estimated Duration:** {script.duration_estimate // 60}m {script.duration_estimate % 60}s\n")
            f.write(f"**Total Lines:** {script.metadata['total_lines']}\n")
            f.write(f"**LLM Provider:** {script.metadata.get('llm_provider', 'unknown')}\n\n")
            f.write("---\n\n")
            
            f.write("## INTRODUCTION\n\n")
            for line in script.intro:
                self._write_line(f, line)
            
            f.write("\n## MAIN CONTENT\n\n")
            for line in script.main_content:
                self._write_line(f, line)
            
            f.write("\n## CONCLUSION\n\n")
            for line in script.outro:
                self._write_line(f, line)
        
        print(f"Saved script to: {output_path}")
    
    def _write_line(self, file, line: DialogueLine):
        """Write a single dialogue line to file"""
        prefix = ""
        if line.emotion:
            prefix += f"[{line.emotion}] "
        if line.action:
            prefix += f"[{line.action}] "
        
        file.write(f"**{line.speaker}:** {prefix}{line.text}\n\n")
    
    def export_for_audio(self, script: PodcastScript, output_path: str):
        """Export script in a format optimized for audio production"""
        lines_data = []
        
        for line in script.get_all_lines():
            lines_data.append({
                'speaker': line.speaker,
                'text': line.text,
                'emotion': line.emotion,
                'action': line.action,
                'timing': line.timing
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'title': script.title,
                'duration_estimate': script.duration_estimate,
                'lines': lines_data,
                'metadata': script.metadata
            }, f, indent=2)
        
        print(f"Exported audio-ready script to: {output_path}")
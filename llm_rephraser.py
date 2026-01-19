"""
LLM Rephraser for Mini Zia
Optional layer to enhance insights with conversational language using LLMs.
"""

import json
from typing import Dict, Optional


class LLMRephraser:
    """
    Enhances rule-based insights with LLM-powered natural language.
    This is a placeholder implementation with optional API integration.
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = 'anthropic'):
        """
        Initialize the LLM rephraser.
        
        Args:
            api_key (str): API key for LLM provider (optional)
            provider (str): LLM provider ('anthropic', 'openai', or 'mock')
        """
        self.api_key = api_key
        self.provider = provider
        self.enabled = api_key is not None
        
        if not self.enabled:
            print("ℹ️  LLM Rephraser initialized in MOCK mode (no API key provided)")
    
    def enhance_insight(self, insight_text: str, 
                       context: Dict = None,
                       tone: str = 'professional') -> str:
        """
        Enhance an insight with more conversational language.
        
        Args:
            insight_text (str): Original rule-based insight
            context (Dict): Additional context for the LLM
            tone (str): Desired tone ('professional', 'casual', 'executive')
        
        Returns:
            str: Enhanced insight
        """
        if not self.enabled:
            return self._mock_enhance(insight_text, tone)
        
        # Placeholder for actual LLM API call
        prompt = self._build_prompt(insight_text, context, tone)
        
        # This is where you would call the actual API
        # For Anthropic Claude:
        # response = anthropic.Anthropic(api_key=self.api_key).messages.create(...)
        
        # For OpenAI:
        # response = openai.ChatCompletion.create(...)
        
        return self._mock_enhance(insight_text, tone)
    
    def _build_prompt(self, insight_text: str, 
                     context: Dict,
                     tone: str) -> str:
        """
        Build the prompt for the LLM.
        
        Args:
            insight_text (str): Original insight
            context (Dict): Additional context
            tone (str): Desired tone
        
        Returns:
            str: Formatted prompt
        """
        tone_instructions = {
            'professional': 'Use professional business language, be clear and concise.',
            'casual': 'Use friendly, approachable language while remaining informative.',
            'executive': 'Use executive-level language, focus on strategic implications.'
        }
        
        prompt = f"""You are a business analyst assistant. Rephrase the following insight 
to make it more {tone} and conversational while preserving all key information.

Tone guidance: {tone_instructions.get(tone, tone_instructions['professional'])}

Original Insight:
{insight_text}

Context:
{json.dumps(context, indent=2) if context else 'No additional context'}

Enhanced Insight:"""
        
        return prompt
    
    def _mock_enhance(self, insight_text: str, tone: str) -> str:
        """
        Mock enhancement for testing without API access.
        
        Args:
            insight_text (str): Original insight
            tone (str): Desired tone
        
        Returns:
            str: Mock enhanced insight
        """
        # Simple rule-based enhancements for demo purposes
        enhancements = {
            'professional': {
                'dropped': 'experienced a decline of',
                'increased': 'showed growth of',
                'Primary cause': 'The main contributing factor is',
                'Requires investigation': 'Warrants immediate attention'
            },
            'casual': {
                'dropped': 'went down by',
                'increased': 'jumped up by',
                'Primary cause': "Here's what's driving this:",
                'Requires investigation': 'Worth taking a closer look'
            },
            'executive': {
                'dropped': 'declined by',
                'increased': 'grew by',
                'Primary cause': 'Strategic driver:',
                'Requires investigation': 'Action required'
            }
        }
        
        enhanced = insight_text
        replacements = enhancements.get(tone, enhancements['professional'])
        
        for original, replacement in replacements.items():
            enhanced = enhanced.replace(original, replacement)
        
        return enhanced
    
    def batch_enhance(self, insights: list, 
                     context: Dict = None,
                     tone: str = 'professional') -> list:
        """
        Enhance multiple insights in batch.
        
        Args:
            insights (list): List of insight strings
            context (Dict): Shared context
            tone (str): Desired tone
        
        Returns:
            list: Enhanced insights
        """
        return [self.enhance_insight(insight, context, tone) for insight in insights]
    
    def explain_metric_change(self, metric: str,
                            change_pct: float,
                            drivers: list,
                            region: str = None) -> str:
        """
        Generate a conversational explanation for metric changes.
        
        Args:
            metric (str): Metric name
            change_pct (float): Percentage change
            drivers (list): List of contributing factors
            region (str): Region name
        
        Returns:
            str: Conversational explanation
        """
        location = f" in {region}" if region else ""
        direction = "increased" if change_pct > 0 else "decreased"
        
        explanation = f"Let me break down what happened with {metric}{location}. "
        explanation += f"We saw a {abs(change_pct):.1f}% {direction}. "
        
        if drivers:
            explanation += "Here are the key factors:\n"
            for i, driver in enumerate(drivers, 1):
                driver_name = driver.get('driver', '').replace('_', ' ').title()
                driver_change = driver.get('change_pct', 0)
                explanation += f"{i}. {driver_name} changed by {driver_change:+.1f}%\n"
        
        return explanation
    
    @staticmethod
    def create_api_integration_guide() -> str:
        """
        Provide guidance on integrating with actual LLM APIs.
        
        Returns:
            str: Integration guide
        """
        guide = """
========================================
LLM API INTEGRATION GUIDE
========================================

To enable actual LLM enhancement, follow these steps:

1. ANTHROPIC CLAUDE INTEGRATION
   
   Install: pip install anthropic
   
   Code:
   ```python
   import anthropic
   
   client = anthropic.Anthropic(api_key="your_api_key")
   message = client.messages.create(
       model="claude-3-5-sonnet-20241022",
       max_tokens=1024,
       messages=[{"role": "user", "content": prompt}]
   )
   enhanced_text = message.content[0].text
   ```

2. OPENAI GPT INTEGRATION
   
   Install: pip install openai
   
   Code:
   ```python
   import openai
   
   openai.api_key = "your_api_key"
   response = openai.ChatCompletion.create(
       model="gpt-4",
       messages=[{"role": "user", "content": prompt}]
   )
   enhanced_text = response.choices[0].message.content
   ```

3. USAGE IN MINI ZIA
   
   ```python
   rephraser = LLMRephraser(api_key="your_key", provider="anthropic")
   enhanced = rephraser.enhance_insight(original_insight)
   ```

4. BEST PRACTICES
   
   - Keep prompts focused on rephrasing, not analysis
   - Preserve all numerical data and facts
   - Set appropriate temperature (0.3-0.5 for consistency)
   - Implement caching for cost optimization
   - Add error handling for API failures

========================================
        """
        return guide


if __name__ == "__main__":
    # Test the LLM rephraser in mock mode
    rephraser = LLMRephraser()
    
    original = """🚨 Sales dropped by 18.5% in Chennai (Enterprise)
   Current: ₹950,000 ↘️
   Priority: CRITICAL
   
🔍 Primary cause in Chennai (Enterprise): Leads declined by 25.0%"""
    
    print("=" * 60)
    print("ORIGINAL INSIGHT")
    print("=" * 60)
    print(original)
    
    print("\n" + "=" * 60)
    print("PROFESSIONAL TONE")
    print("=" * 60)
    print(rephraser.enhance_insight(original, tone='professional'))
    
    print("\n" + "=" * 60)
    print("CASUAL TONE")
    print("=" * 60)
    print(rephraser.enhance_insight(original, tone='casual'))
    
    print("\n" + "=" * 60)
    print("EXECUTIVE TONE")
    print("=" * 60)
    print(rephraser.enhance_insight(original, tone='executive'))
    
    print("\n")
    print(LLMRephraser.create_api_integration_guide())
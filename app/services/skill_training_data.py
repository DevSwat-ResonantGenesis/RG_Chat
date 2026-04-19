"""
Skill Router Training Data
============================

High-quality labeled examples for training the neural skill classifier.
Each example is a (message, context, correct_skill) triple.

Context format: list of recent messages as dicts with "role" and "content".
Skill "none" means general chat — no tool needed.

This is the SEED dataset. Real user interactions are collected via active
learning and merged during periodic retraining.
"""
from typing import List, Dict, Any, Optional, Tuple

# Type alias for training samples
# (user_message, recent_context, correct_skill_id_or_none)
TrainingSample = Tuple[str, List[Dict[str, str]], Optional[str]]


def get_training_data() -> List[TrainingSample]:
    """Return the full seed training dataset."""
    samples: List[TrainingSample] = []

    # ------------------------------------------------------------------
    # AGENT_ARCHITECT — building, managing, running agents
    # ------------------------------------------------------------------
    _arch = "agent_architect"

    # Direct requests
    samples += [
        ("build me an agent that monitors stock prices every hour", [], _arch),
        ("create an agent to scrape Y Combinator and list startups", [], _arch),
        ("I want to automate scraping websites and saving results", [], _arch),
        ("make an agent that checks my competitors' pricing daily", [], _arch),
        ("set up an autonomous agent to collect news about AI", [], _arch),
        ("build a bot that searches Twitter for mentions of my brand", [], _arch),
        ("create an agent to monitor Hacker News for relevant posts", [], _arch),
        ("I need an agent that summarizes my emails every morning", [], _arch),
        ("how many agents do I have", [], _arch),
        ("show my agents", [], _arch),
        ("list all my agents", [], _arch),
        ("delete the test agent", [], _arch),
        ("rename my scraper agent to DataCollector", [], _arch),
        ("run my weather agent now", [], _arch),
        ("stop the news monitoring agent", [], _arch),
        ("schedule the report agent to run every day at 9am", [], _arch),
        ("modify my scraper to also save results to Google Drive", [], _arch),
        ("my agent keeps failing, can you diagnose it", [], _arch),
        ("the web scraper agent is broken", [], _arch),
        ("configure the agent to use GPT-4 instead of Gemini", [], _arch),
        ("add web search tool to my research agent", [], _arch),
        ("I want to build something that automatically researches topics", [], _arch),
        ("can you make me an automated workflow to process data", [], _arch),
        ("create an AI assistant that handles customer questions", [], _arch),
        ("build an agent to track cryptocurrency prices", [], _arch),
        ("set up a monitoring agent for my website uptime", [], _arch),
        ("I want an agent that reads RSS feeds and sends summaries", [], _arch),
        ("create a data collection agent for market research", [], _arch),
        ("build me an agent", [], _arch),
        ("I want to create an autonomous bot", [], _arch),
        ("make an automated pipeline for web scraping", [], _arch),
        ("help me set up an agent to do recurring tasks", [], _arch),
    ]

    # Follow-ups in agent architect context
    samples += [
        ("yes, create it", [
            {"role": "assistant", "content": "I'll build an agent that scrapes Y Combinator. The agent will use web_search and fetch_url tools. Shall I proceed?"},
        ], _arch),
        ("use Gemini for the LLM", [
            {"role": "assistant", "content": "What LLM provider would you like for this agent? Options: GPT-4, Gemini, Claude"},
        ], _arch),
        ("yes create all of them", [
            {"role": "assistant", "content": "I've planned 3 agents: News Monitor, Price Tracker, Report Generator. Want me to create all three?"},
        ], _arch),
        ("try again", [
            {"role": "assistant", "content": "The agent creation failed because web_search tool returned an error. Would you like me to retry?"},
        ], _arch),
        ("add Google Drive to it", [
            {"role": "assistant", "content": "The agent is ready. It will scrape websites and collect data. Any additional tools you'd like?"},
        ], _arch),
        ("change the schedule to every 2 hours", [
            {"role": "assistant", "content": "The agent is scheduled to run every hour. Want to modify the schedule?"},
        ], _arch),
        ("yes please proceed", [
            {"role": "user", "content": "build me an agent that tracks bitcoin price"},
            {"role": "assistant", "content": "I'll create a Bitcoin Price Tracker agent. It will check the price every 30 minutes. Ready to build?"},
        ], _arch),
        ("actually make it check every 15 minutes", [
            {"role": "user", "content": "build an agent to monitor prices"},
            {"role": "assistant", "content": "How often should it check? I recommend every 30 minutes to balance freshness and credits."},
        ], _arch),
        ("no, use web search instead of fetch_url", [
            {"role": "assistant", "content": "For the scraping agent, I'll configure these tools: fetch_url, code_sandbox. Sound good?"},
        ], _arch),
    ]

    # ------------------------------------------------------------------
    # GOOGLE_DRIVE — file access, search, create
    # ------------------------------------------------------------------
    _gdrive = "google_drive"

    samples += [
        ("search my Google Drive for the quarterly report", [], _gdrive),
        ("find documents about the project in my Drive", [], _gdrive),
        ("upload this summary to Google Drive", [], _gdrive),
        ("create a spreadsheet in my Drive with these results", [], _gdrive),
        ("list my recent Drive files", [], _gdrive),
        ("open the marketing plan from my Drive", [], _gdrive),
        ("save this to my Google Drive as an Excel file", [], _gdrive),
        ("what files do I have in Drive", [], _gdrive),
        ("check my Drive for the meeting notes", [], _gdrive),
        ("download the budget spreadsheet from Drive", [], _gdrive),
    ]

    # Drive follow-ups
    samples += [
        ("yes save it as Excel", [
            {"role": "assistant", "content": "I can save the summary to your Google Drive. What format would you prefer — CSV, Excel, or Google Sheets?"},
        ], _gdrive),
        ("the marketing folder", [
            {"role": "assistant", "content": "Where in your Drive should I save the file?"},
        ], _gdrive),
    ]

    # ------------------------------------------------------------------
    # GOOGLE_CALENDAR — events, schedule, meetings
    # ------------------------------------------------------------------
    _gcal = "google_calendar"

    samples += [
        ("what's on my calendar today", [], _gcal),
        ("schedule a meeting for tomorrow at 3pm", [], _gcal),
        ("create a team standup event for Monday", [], _gcal),
        ("show my upcoming events this week", [], _gcal),
        ("when is my next meeting", [], _gcal),
        ("add a reminder to my calendar for Friday", [], _gcal),
        ("block 2pm to 4pm on my calendar", [], _gcal),
        ("do I have anything scheduled for Wednesday", [], _gcal),
        ("cancel my 10am meeting tomorrow", [], _gcal),
        ("move the design review to Thursday", [], _gcal),
    ]

    # ------------------------------------------------------------------
    # WEB_SEARCH — real-time info, news, prices
    # ------------------------------------------------------------------
    _web = "web_search"

    samples += [
        ("what's the weather in San Francisco right now", [], _web),
        ("latest news about artificial intelligence", [], _web),
        ("current Bitcoin price", [], _web),
        ("who won the NBA game last night", [], _web),
        ("what's trending on Twitter today", [], _web),
        ("search for recent papers on transformer models", [], _web),
        ("what happened in the stock market today", [], _web),
        ("find the latest SpaceX launch news", [], _web),
        ("current price of Tesla stock", [], _web),
        ("search for new AI startups in 2026", [], _web),
        ("what's the latest on the Ukraine situation", [], _web),
        ("look up recent developments in quantum computing", [], _web),
        ("find reviews for the new iPhone", [], _web),
        ("what are today's top headlines", [], _web),
        ("search for best restaurants in downtown LA", [], _web),
    ]

    # ------------------------------------------------------------------
    # IMAGE_GENERATION — create/draw/generate images
    # ------------------------------------------------------------------
    _img = "image_generation"

    samples += [
        ("generate an image of a sunset over mountains", [], _img),
        ("create a logo for my startup called NeuralFlow", [], _img),
        ("draw a cute robot holding a flower", [], _img),
        ("make me a picture of a futuristic city at night", [], _img),
        ("illustrate a dragon flying over a medieval castle", [], _img),
        ("generate a portrait in watercolor style", [], _img),
        ("create an abstract art piece with blue and gold", [], _img),
        ("draw a cyberpunk street scene", [], _img),
        ("make an image of a cozy coffee shop interior", [], _img),
        ("generate a product mockup for a mobile app", [], _img),
    ]

    # ------------------------------------------------------------------
    # CODE_VISUALIZER — GitHub repo scanning
    # ------------------------------------------------------------------
    _cv = "code_visualizer"

    samples += [
        ("scan this repo https://github.com/facebook/react", [], _cv),
        ("analyze my GitHub repository https://github.com/user/project", [], _cv),
        ("visualize the codebase structure of https://github.com/vercel/next.js", [], _cv),
        ("scan https://github.com/tensorflow/tensorflow", [], _cv),
        ("analyze the architecture of this GitHub repo https://github.com/openai/whisper", [], _cv),
        ("show me the code structure of https://github.com/langchain-ai/langchain", [], _cv),
        ("scan my project on GitHub and show the dependencies", [], _cv),
        ("visualize https://github.com/microsoft/vscode", [], _cv),
    ]

    # ------------------------------------------------------------------
    # MEMORY_SEARCH — recall past conversations
    # ------------------------------------------------------------------
    _mem = "memory_search"

    samples += [
        ("what did I say about the database last week", [], _mem),
        ("do you remember our conversation about React", [], _mem),
        ("recall what we discussed about the API design", [], _mem),
        ("search my memory for deployment instructions", [], _mem),
        ("what did we talk about yesterday", [], _mem),
        ("find our previous discussion about pricing", [], _mem),
        ("what was that thing I told you about the project structure", [], _mem),
        ("do you remember the password I mentioned", [], _mem),
        ("recall the architecture decisions we made", [], _mem),
    ]

    # ------------------------------------------------------------------
    # MEMORY_LIBRARY — open panel
    # ------------------------------------------------------------------
    _memlib = "memory_library"

    samples += [
        ("open memory library", [], _memlib),
        ("show my memories", [], _memlib),
        ("browse memories", [], _memlib),
        ("open the memory panel", [], _memlib),
    ]

    # ------------------------------------------------------------------
    # STATE_PHYSICS — panel
    # ------------------------------------------------------------------
    _sp = "state_physics"

    samples += [
        ("open state physics", [], _sp),
        ("show state physics visualization", [], _sp),
        ("state-space visualization", [], _sp),
    ]

    # ------------------------------------------------------------------
    # IDE_WORKSPACE — panel
    # ------------------------------------------------------------------
    _ide = "ide_workspace"

    samples += [
        ("open IDE", [], _ide),
        ("open the editor", [], _ide),
        ("open terminal", [], _ide),
        ("open workspace", [], _ide),
        ("launch the code editor", [], _ide),
    ]

    # ------------------------------------------------------------------
    # RABBIT_POST — community
    # ------------------------------------------------------------------
    _rabbit = "rabbit_post"

    samples += [
        ("post this to Rabbit community", [], _rabbit),
        ("share this on the forum", [], _rabbit),
        ("create a Rabbit post about my project", [], _rabbit),
        ("publish to Rabbit", [], _rabbit),
    ]

    # ------------------------------------------------------------------
    # FIGMA — designs
    # ------------------------------------------------------------------
    _figma = "figma"

    samples += [
        ("show my Figma projects", [], _figma),
        ("open my Figma designs", [], _figma),
        ("list Figma components in the dashboard project", [], _figma),
        ("search my Figma files for the login screen", [], _figma),
    ]

    # ------------------------------------------------------------------
    # SIGMA — analytics
    # ------------------------------------------------------------------
    _sigma = "sigma"

    samples += [
        ("show my Sigma dashboards", [], _sigma),
        ("open the Sigma analytics report", [], _sigma),
        ("get Sigma data for Q1", [], _sigma),
    ]

    # ------------------------------------------------------------------
    # NONE — general chat, no tool needed (CRITICAL: must be well-represented)
    # ------------------------------------------------------------------
    _none = None

    samples += [
        # General conversation
        ("hello", [], _none),
        ("hi there", [], _none),
        ("how are you", [], _none),
        ("thanks", [], _none),
        ("thank you so much", [], _none),
        ("goodbye", [], _none),
        ("what can you do", [], _none),
        ("who are you", [], _none),
        ("tell me about yourself", [], _none),
        ("that's cool", [], _none),
        ("nice", [], _none),
        ("ok", [], _none),
        ("got it", [], _none),
        ("sure", [], _none),
        ("interesting", [], _none),

        # Knowledge questions (AI can answer from training)
        ("what is machine learning", [], _none),
        ("explain how neural networks work", [], _none),
        ("what is the difference between Python and JavaScript", [], _none),
        ("how does a database work", [], _none),
        ("explain quantum computing in simple terms", [], _none),
        ("what is the theory of relativity", [], _none),
        ("how do transformers work in NLP", [], _none),
        ("what is a REST API", [], _none),
        ("explain Docker containers", [], _none),
        ("what is Kubernetes", [], _none),
        ("how does blockchain work", [], _none),
        ("what is the capital of France", [], _none),
        ("who invented the internet", [], _none),
        ("explain the water cycle", [], _none),
        ("what are design patterns in software", [], _none),

        # Coding help (no tool needed)
        ("help me write a Python function to sort a list", [], _none),
        ("how do I fix this TypeError in JavaScript", [], _none),
        ("write a React component for a login form", [], _none),
        ("explain this SQL query", [], _none),
        ("debug this code for me", [], _none),
        ("convert this Python code to JavaScript", [], _none),
        ("optimize this algorithm", [], _none),
        ("what's wrong with this function", [], _none),
        ("help me with CSS flexbox", [], _none),
        ("write a bash script to backup files", [], _none),
        ("refactor this class to be more efficient", [], _none),
        ("explain the difference between async and sync", [], _none),
        ("how do I handle errors in Python", [], _none),
        ("write unit tests for this function", [], _none),

        # Math
        ("what is 234 times 567", [], _none),
        ("solve this equation: 2x + 5 = 15", [], _none),
        ("calculate the compound interest on $10,000", [], _none),

        # Opinions/advice
        ("what do you think about React vs Vue", [], _none),
        ("should I use PostgreSQL or MongoDB", [], _none),
        ("is it better to learn Python or JavaScript first", [], _none),
        ("what's the best way to learn machine learning", [], _none),

        # Meta questions about the platform (not needing tools)
        ("how does this platform work", [], _none),
        ("what features do you have", [], _none),
        ("what's your pricing", [], _none),
        ("how do credits work", [], _none),

        # Ambiguous but NOT tool-related
        ("I'm working on a project", [], _none),
        ("I need some help", [], _none),
        ("can you assist me", [], _none),
        ("I have a question", [], _none),
        ("let me think about that", [], _none),

        # Tricky: mentions "agent" but NOT wanting to build one
        ("what is an AI agent", [], _none),
        ("explain how agents work in reinforcement learning", [], _none),
        ("what's the difference between an agent and a bot", [], _none),
        ("how do autonomous agents make decisions", [], _none),

        # Tricky: mentions "search" but NOT web search
        ("search through this code for bugs", [], _none),
        ("search the documentation for authentication", [], _none),

        # Tricky: mentions "image" but NOT generation
        ("what does this image show", [], _none),
        ("explain this screenshot", [], _none),
        ("analyze the image I uploaded", [], _none),

        # Tricky: mentions "drive" but NOT Google Drive
        ("what drives customer behavior", [], _none),
        ("the main drive behind this project is innovation", [], _none),

        # Tricky: mentions "calendar" but NOT Google Calendar
        ("what calendar year are we in", [], _none),
        ("the fiscal calendar starts in April", [], _none),

        # Tricky: mentions "code" but NOT code_visualizer
        ("write me some code", [], _none),
        ("help me understand this code", [], _none),
        ("review my code", [], _none),
        ("what's the best code editor", [], _none),
    ]

    # ------------------------------------------------------------------
    # CROSS-SKILL CONTINUITY — follow-ups that should stick with active skill
    # ------------------------------------------------------------------

    # User answering architect's questions (mentions Drive but should stay with architect)
    samples += [
        ("yes I have a Google Drive account, and Excel format please", [
            {"role": "user", "content": "build an agent that scrapes websites and saves to Drive"},
            {"role": "assistant", "content": "Do you have a Google Drive account? What format for the summary sheet — CSV or Excel?"},
        ], _arch),
        ("I just connected it, try again", [
            {"role": "assistant", "content": "Google Drive is not connected. Go to Settings → Connect Profiles to add your API key."},
            {"role": "user", "content": "build an agent to save data to my drive"},
        ], _arch),
        ("excel format", [
            {"role": "user", "content": "create an agent that monitors prices and exports to sheets"},
            {"role": "assistant", "content": "What format would you like for the export — CSV, Excel, or Google Sheets?"},
        ], _arch),
        ("ye", [
            {"role": "user", "content": "build me an agent to track news"},
            {"role": "assistant", "content": "I'll create a News Tracker agent with web_search and summarization. Proceed?"},
        ], _arch),
        ("no change it to every 2 hours", [
            {"role": "user", "content": "create an agent to check prices"},
            {"role": "assistant", "content": "The agent is set to run every hour. Ready to deploy?"},
        ], _arch),
    ]

    # User answering Google Drive's questions (should stay with Drive)
    samples += [
        ("the reports folder", [
            {"role": "assistant", "content": "Where in your Google Drive should I save this file?"},
        ], _gdrive),
        ("csv format please", [
            {"role": "assistant", "content": "I can save this data to your Drive. What format — CSV, Excel, or Sheets?"},
        ], _gdrive),
    ]

    # General follow-ups in general chat context (should stay None)
    samples += [
        ("yes", [
            {"role": "assistant", "content": "Machine learning is a subset of AI that learns from data. Want me to explain specific algorithms?"},
        ], _none),
        ("tell me more", [
            {"role": "assistant", "content": "Transformers use self-attention mechanisms to process sequences in parallel."},
        ], _none),
        ("can you elaborate", [
            {"role": "assistant", "content": "Docker containers package applications with their dependencies for consistent deployment."},
        ], _none),
        ("thanks that helps", [
            {"role": "assistant", "content": "Here's the refactored version of your function with better error handling."},
        ], _none),
    ]

    return samples

"""
Agent Classifier Training Data
================================

High-quality labeled examples for training the neural agent classifier.
Each example is a (message, correct_agent_type) pair.

Agent types match AGENT_CAPABILITIES in agent_capability_registry.py.
This is the SEED dataset — active learning samples accumulate in DB.
"""
from typing import List, Tuple

# (user_message, agent_type)
TrainingSample = Tuple[str, str]


def get_agent_training_data() -> List[TrainingSample]:
    """Return the full seed training dataset for agent classification."""
    samples: List[TrainingSample] = []

    # ------------------------------------------------------------------
    # REASONING — analysis, logic, problem solving, general questions
    # ------------------------------------------------------------------
    samples += [
        ("analyze the pros and cons of microservices vs monolith", "reasoning"),
        ("explain why this approach might fail at scale", "reasoning"),
        ("what causes memory leaks in Node.js", "reasoning"),
        ("how does the garbage collector work in Python", "reasoning"),
        ("think through the tradeoffs of using Redis vs PostgreSQL", "reasoning"),
        ("what are the implications of this design decision", "reasoning"),
        ("why would someone choose GraphQL over REST", "reasoning"),
        ("compare event sourcing and CQRS", "reasoning"),
        ("what's the reasoning behind immutable data structures", "reasoning"),
        ("help me understand the CAP theorem", "reasoning"),
        ("what are the tradeoffs of eventual consistency", "reasoning"),
        ("analyze this system for potential bottlenecks", "reasoning"),
        ("what would happen if we removed this dependency", "reasoning"),
        ("I need to think through a complex decision", "reasoning"),
        ("help me reason about the best approach", "reasoning"),
    ]

    # ------------------------------------------------------------------
    # CODE — code generation, implementation, programming
    # ------------------------------------------------------------------
    samples += [
        ("write a Python function to merge two sorted arrays", "code"),
        ("generate code for a REST API in Express", "code"),
        ("create a React component for a login form", "code"),
        ("implement a binary search tree in TypeScript", "code"),
        ("write a script to parse CSV files", "code"),
        ("code a websocket server in Python", "code"),
        ("implement a middleware for authentication in Express", "code"),
        ("write the database migration for adding a users table", "code"),
        ("generate a Docker Compose file for my stack", "code"),
        ("create a utility function for deep cloning objects", "code"),
        ("implement pagination for my API", "code"),
        ("write a class for managing database connections", "code"),
        ("code me a debounce function in JavaScript", "code"),
        ("build a simple CLI tool in Python", "code"),
        ("implement rate limiting middleware", "code"),
    ]

    # ------------------------------------------------------------------
    # DEBUG — fixing issues, troubleshooting, errors
    # ------------------------------------------------------------------
    samples += [
        ("fix this error: TypeError cannot read property of undefined", "debug"),
        ("debug why my API returns 500 on POST requests", "debug"),
        ("my function throws an exception when input is null", "debug"),
        ("why is this SQL query so slow", "debug"),
        ("the Docker container keeps crashing on startup", "debug"),
        ("my tests are failing with timeout errors", "debug"),
        ("find the bug in this sorting algorithm", "debug"),
        ("this endpoint works locally but not in production", "debug"),
        ("help me troubleshoot this CORS issue", "debug"),
        ("the WebSocket connection keeps dropping", "debug"),
        ("fix the memory leak in this component", "debug"),
        ("my build is broken after upgrading dependencies", "debug"),
        ("why does this regex not match my input", "debug"),
        ("something is wrong with my authentication flow", "debug"),
        ("this code isn't working as expected", "debug"),
    ]

    # ------------------------------------------------------------------
    # REVIEW — code review, quality assessment, feedback
    # ------------------------------------------------------------------
    samples += [
        ("review this pull request for best practices", "review"),
        ("critique my API design", "review"),
        ("give feedback on this code structure", "review"),
        ("check this function for edge cases", "review"),
        ("review my database schema for normalization issues", "review"),
        ("assess the quality of this implementation", "review"),
        ("what could be improved in this code", "review"),
        ("is this a good approach for handling errors", "review"),
        ("review my authentication implementation", "review"),
        ("evaluate this algorithm for efficiency", "review"),
    ]

    # ------------------------------------------------------------------
    # TEST — test generation, coverage, testing strategies
    # ------------------------------------------------------------------
    samples += [
        ("write unit tests for this function", "test"),
        ("generate test cases for the login endpoint", "test"),
        ("what edge cases should I test for this validator", "test"),
        ("create integration tests for the checkout flow", "test"),
        ("set up Jest testing for my React components", "test"),
        ("write pytest fixtures for my database tests", "test"),
        ("generate a test strategy for this feature", "test"),
        ("add test coverage for error handling paths", "test"),
        ("create mock data for my API tests", "test"),
        ("write end-to-end tests with Playwright", "test"),
    ]

    # ------------------------------------------------------------------
    # MATH — calculations, equations, formulas
    # ------------------------------------------------------------------
    samples += [
        ("calculate the compound interest on $10,000 at 5% for 10 years", "math"),
        ("solve this quadratic equation: x² + 5x + 6 = 0", "math"),
        ("what's the derivative of sin(x) * cos(x)", "math"),
        ("compute the eigenvalues of this matrix", "math"),
        ("calculate the standard deviation of this dataset", "math"),
        ("solve the integral of e^x * sin(x)", "math"),
        ("what's the probability of rolling three sixes in a row", "math"),
        ("convert 150 miles to kilometers", "math"),
        ("calculate the time complexity of this algorithm", "math"),
        ("find the area under the curve y = x^3 from 0 to 2", "math"),
        ("what's 15% of 2,340", "math"),
        ("solve this system of equations", "math"),
    ]

    # ------------------------------------------------------------------
    # RESEARCH — information gathering, investigation, comparison
    # ------------------------------------------------------------------
    samples += [
        ("research the best practices for microservices authentication", "research"),
        ("find information about WebAssembly performance benchmarks", "research"),
        ("investigate the different state management solutions for React", "research"),
        ("compare Kubernetes vs Docker Swarm for container orchestration", "research"),
        ("what are the latest trends in serverless computing", "research"),
        ("look up the recommended security headers for web apps", "research"),
        ("research how to implement OAuth 2.0 with PKCE", "research"),
        ("find the best database for time-series data", "research"),
        ("investigate the pros and cons of each cloud provider", "research"),
        ("compare different message queue systems", "research"),
    ]

    # ------------------------------------------------------------------
    # SUMMARY — condensing information, key points, brevity
    # ------------------------------------------------------------------
    samples += [
        ("summarize our conversation so far", "summary"),
        ("give me a brief overview of this project", "summary"),
        ("tl;dr of the React documentation", "summary"),
        ("what are the key points from this article", "summary"),
        ("summarize the main differences between TCP and UDP", "summary"),
        ("give me a quick summary of these changes", "summary"),
        ("condense this long explanation into bullet points", "summary"),
        ("what's the gist of this design document", "summary"),
        ("brief overview of GraphQL subscriptions", "summary"),
        ("summarize the meeting notes", "summary"),
    ]

    # ------------------------------------------------------------------
    # PLANNING — strategy, roadmaps, step-by-step organization
    # ------------------------------------------------------------------
    samples += [
        ("create a project plan for building a SaaS product", "planning"),
        ("help me plan the migration from MySQL to PostgreSQL", "planning"),
        ("what steps do I need to take to deploy to production", "planning"),
        ("create a roadmap for implementing CI/CD", "planning"),
        ("plan the architecture for a real-time chat application", "planning"),
        ("how should I organize my monorepo", "planning"),
        ("help me create a sprint plan for this feature", "planning"),
        ("what's the strategy for scaling our database", "planning"),
        ("plan the rollout for the new API version", "planning"),
        ("create a learning path for becoming a DevOps engineer", "planning"),
    ]

    # ------------------------------------------------------------------
    # SECURITY — vulnerability detection, security analysis
    # ------------------------------------------------------------------
    samples += [
        ("check this code for SQL injection vulnerabilities", "security"),
        ("audit the authentication flow for security issues", "security"),
        ("is this implementation vulnerable to XSS attacks", "security"),
        ("review the CSRF protection in my app", "security"),
        ("analyze the encryption scheme for weaknesses", "security"),
        ("what security headers should I add", "security"),
        ("check for insecure deserialization vulnerabilities", "security"),
        ("is this JWT implementation secure", "security"),
        ("analyze the threat model for this API", "security"),
        ("review the password hashing implementation", "security"),
    ]

    # ------------------------------------------------------------------
    # ARCHITECTURE — system design, scalability, high-level design
    # NOTE: "agent" + management verbs must NOT land here. Those go to
    #       agent_architect TOOL via the ToolClassifier, not this agent.
    # ------------------------------------------------------------------
    samples += [
        ("design a scalable notification system", "architecture"),
        ("what's the best architecture for a real-time analytics platform", "architecture"),
        ("how should I structure my microservices", "architecture"),
        ("design the database schema for an e-commerce platform", "architecture"),
        ("propose an architecture for handling millions of events per second", "architecture"),
        ("how to architect a multi-tenant SaaS application", "architecture"),
        ("design a fault-tolerant message processing system", "architecture"),
        ("what design patterns should I use for this system", "architecture"),
        ("architect a CDN for global content delivery", "architecture"),
        ("design a distributed caching layer", "architecture"),
    ]

    # Anti-collision: agent management queries must NOT map to architecture.
    # These should be handled by agent_architect TOOL. If the tool classifier
    # misses them and they land here, route to "reasoning" (safe default).
    samples += [
        ("create an agent for me", "reasoning"),
        ("build me an agent", "reasoning"),
        ("list my agents", "reasoning"),
        ("show my agents", "reasoning"),
        ("how many agents do I have", "reasoning"),
        ("run my agent", "reasoning"),
        ("stop the agent", "reasoning"),
        ("delete my agent", "reasoning"),
        ("configure my agent", "reasoning"),
        ("agent status", "reasoning"),
        ("check my agents", "reasoning"),
        ("design an agent for web scraping", "reasoning"),
        ("architect an agent system", "reasoning"),
        ("help me design my agent", "reasoning"),
    ]

    # ------------------------------------------------------------------
    # EXPLAIN — simplified explanations, teaching, ELI5
    # ------------------------------------------------------------------
    samples += [
        ("explain Docker containers like I'm five", "explain"),
        ("what is Kubernetes in simple terms", "explain"),
        ("explain async/await to a beginner", "explain"),
        ("what are design patterns explained simply", "explain"),
        ("help me understand recursion with examples", "explain"),
        ("explain the basics of HTTP", "explain"),
        ("what is a database index in layman's terms", "explain"),
        ("teach me about REST APIs from scratch", "explain"),
        ("explain blockchain technology simply", "explain"),
        ("introduction to machine learning concepts", "explain"),
    ]

    # ------------------------------------------------------------------
    # OPTIMIZATION — performance, efficiency, bottlenecks
    # ------------------------------------------------------------------
    samples += [
        ("optimize this SQL query for performance", "optimization"),
        ("how to speed up my React app rendering", "optimization"),
        ("find the bottleneck in this data pipeline", "optimization"),
        ("optimize the memory usage of this function", "optimization"),
        ("make this algorithm faster", "optimization"),
        ("reduce the bundle size of my web app", "optimization"),
        ("optimize database connection pooling", "optimization"),
        ("how to make this API endpoint more efficient", "optimization"),
        ("improve the load time of my website", "optimization"),
        ("optimize the Docker image size", "optimization"),
    ]

    # ------------------------------------------------------------------
    # DOCUMENTATION — docs generation, API docs, technical writing
    # ------------------------------------------------------------------
    samples += [
        ("write a README for this project", "documentation"),
        ("generate JSDoc comments for these functions", "documentation"),
        ("create API documentation for my endpoints", "documentation"),
        ("write docstrings for this Python module", "documentation"),
        ("document the deployment process", "documentation"),
        ("create a changelog for this release", "documentation"),
        ("write technical documentation for the architecture", "documentation"),
        ("generate OpenAPI spec for my REST API", "documentation"),
        ("document the database schema", "documentation"),
        ("write inline comments explaining this complex logic", "documentation"),
    ]

    # ------------------------------------------------------------------
    # REFACTOR — code restructuring, clean code
    # ------------------------------------------------------------------
    samples += [
        ("refactor this function to use async/await", "refactor"),
        ("restructure this monolithic file into modules", "refactor"),
        ("clean up this spaghetti code", "refactor"),
        ("apply the strategy pattern to this code", "refactor"),
        ("simplify this deeply nested conditional", "refactor"),
        ("extract this repeated logic into a shared utility", "refactor"),
        ("convert this class to use dependency injection", "refactor"),
        ("reorganize the project structure", "refactor"),
        ("decouple these tightly coupled modules", "refactor"),
        ("refactor to remove code duplication", "refactor"),
    ]

    # ------------------------------------------------------------------
    # MIGRATION — upgrades, transitions, conversions
    # ------------------------------------------------------------------
    samples += [
        ("migrate from JavaScript to TypeScript", "migration"),
        ("upgrade from React 17 to React 18", "migration"),
        ("convert this Python 2 code to Python 3", "migration"),
        ("port this Express app to Fastify", "migration"),
        ("transition from REST to GraphQL", "migration"),
        ("migrate the database from MySQL to PostgreSQL", "migration"),
        ("move from Webpack to Vite", "migration"),
        ("switch from class components to functional components", "migration"),
        ("upgrade Node.js from version 16 to 20", "migration"),
        ("migrate from monolith to microservices", "migration"),
    ]

    # ------------------------------------------------------------------
    # API — API design, endpoints, REST, GraphQL
    # ------------------------------------------------------------------
    samples += [
        ("design a RESTful API for user management", "api"),
        ("create GraphQL resolvers for the product catalog", "api"),
        ("implement rate limiting for my API endpoints", "api"),
        ("design the webhook system for my platform", "api"),
        ("create API routes for the payment flow", "api"),
        ("design the API versioning strategy", "api"),
        ("implement HATEOAS for my REST API", "api"),
        ("create an API gateway configuration", "api"),
        ("design the authentication API endpoints", "api"),
        ("implement pagination for the list endpoints", "api"),
    ]

    # ------------------------------------------------------------------
    # DATABASE — SQL, queries, schema design
    # ------------------------------------------------------------------
    samples += [
        ("write an SQL query to find duplicate records", "database"),
        ("design the schema for a social media app", "database"),
        ("create an index strategy for this table", "database"),
        ("write a migration to add a new column", "database"),
        ("optimize this PostgreSQL query with JOINs", "database"),
        ("set up MongoDB aggregation pipeline", "database"),
        ("design the Redis caching strategy", "database"),
        ("write a stored procedure for batch updates", "database"),
        ("normalize this database schema to 3NF", "database"),
        ("create a query for time-series data analysis", "database"),
    ]

    # ------------------------------------------------------------------
    # DEVOPS — deployment, CI/CD, Docker, Kubernetes
    # ------------------------------------------------------------------
    samples += [
        ("set up a CI/CD pipeline with GitHub Actions", "devops"),
        ("write a Dockerfile for my Python application", "devops"),
        ("create Kubernetes deployment manifests", "devops"),
        ("configure Nginx as a reverse proxy", "devops"),
        ("set up monitoring with Prometheus and Grafana", "devops"),
        ("deploy to AWS using Terraform", "devops"),
        ("create a Docker Compose for local development", "devops"),
        ("configure auto-scaling for my ECS service", "devops"),
        ("set up a production deployment pipeline", "devops"),
        ("configure SSL certificates with Let's Encrypt", "devops"),
    ]

    # ------------------------------------------------------------------
    # ACCESSIBILITY — a11y, WCAG, ARIA
    # ------------------------------------------------------------------
    samples += [
        ("check this form for accessibility compliance", "accessibility"),
        ("add ARIA labels to my navigation component", "accessibility"),
        ("make this modal keyboard accessible", "accessibility"),
        ("check WCAG 2.1 compliance for this page", "accessibility"),
        ("improve screen reader support for this table", "accessibility"),
        ("fix the color contrast issues in my design", "accessibility"),
        ("add proper alt text to all images", "accessibility"),
        ("make the dropdown menu accessible", "accessibility"),
    ]

    # ------------------------------------------------------------------
    # I18N — internationalization, localization, translation
    # ------------------------------------------------------------------
    samples += [
        ("set up internationalization for my React app", "i18n"),
        ("implement multi-language support with i18next", "i18n"),
        ("create translation files for Spanish and French", "i18n"),
        ("add RTL support for Arabic", "i18n"),
        ("configure locale-aware date formatting", "i18n"),
        ("extract all hardcoded strings for translation", "i18n"),
        ("implement language detection and switching", "i18n"),
        ("localize number and currency formatting", "i18n"),
    ]

    # ------------------------------------------------------------------
    # REGEX — regular expressions, pattern matching
    # ------------------------------------------------------------------
    samples += [
        ("write a regex to validate email addresses", "regex"),
        ("create a regular expression for phone numbers", "regex"),
        ("match URLs with optional query parameters", "regex"),
        ("regex to extract dates in MM/DD/YYYY format", "regex"),
        ("pattern to match IPv4 and IPv6 addresses", "regex"),
        ("write a regex for password validation", "regex"),
        ("extract all HTML tags from a string", "regex"),
        ("regex to match markdown links", "regex"),
    ]

    # ------------------------------------------------------------------
    # GIT — version control, merge conflicts, branching
    # ------------------------------------------------------------------
    samples += [
        ("help me resolve this merge conflict", "git"),
        ("what's the best git branching strategy for a team", "git"),
        ("how to rebase my feature branch onto main", "git"),
        ("cherry-pick a commit from another branch", "git"),
        ("undo the last commit but keep changes", "git"),
        ("squash my last 5 commits into one", "git"),
        ("set up git hooks for pre-commit linting", "git"),
        ("recover a deleted branch in git", "git"),
    ]

    # ------------------------------------------------------------------
    # CSS — styling, flexbox, grid, responsive design
    # ------------------------------------------------------------------
    samples += [
        ("center a div vertically and horizontally with flexbox", "css"),
        ("create a responsive grid layout", "css"),
        ("add a hover animation to this button", "css"),
        ("implement a sticky header with CSS", "css"),
        ("create a dark mode theme with CSS variables", "css"),
        ("make this layout responsive for mobile", "css"),
        ("style a custom dropdown select", "css"),
        ("implement a CSS-only accordion", "css"),
        ("fix the z-index stacking issue", "css"),
        ("create a Tailwind component for a card", "css"),
    ]

    return samples

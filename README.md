# 🍳 MasterChef AI Chatbot

A sophisticated AI-powered recipe chatbot built with **LangGraph**, **LangChain**, and **Streamlit**. This application provides professional cooking guidance, detailed recipes, and culinary expertise through an intelligent conversational interface.

## 🎯 Project Overview

The MasterChef AI Chatbot is a multi-turn conversational application that intelligently validates user queries, determines if they're recipe-related, and provides expert-level cooking assistance. It uses a state-based workflow architecture to ensure reliable query processing and provides a beautiful, responsive UI for seamless user interaction.

### Key Features

- **🤖 Intelligent Query Validation**: Automatically detects recipe-related queries using LLM-based classification
- **📚 Detailed Recipe Generation**: Provides comprehensive recipes with ingredients, step-by-step instructions, tips, and serving suggestions
- **💬 Multi-Turn Conversations**: Maintains conversation history and context across multiple user interactions
- **📊 Session Management**: SQLite-backed persistent storage for chat sessions and message history
- **🎨 Beautiful UI**: Custom-styled Streamlit interface with animated title, chat bubbles, and markdown rendering
- **🔄 Conditional Workflow**: Advanced LangGraph state machine for intelligent request routing
- **💾 Data Persistence**: All conversations stored in SQLite database for future reference

## 🏗️ Architecture

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | OpenAI GPT-4o Mini |
| **Orchestration** | LangGraph (State Graph) |
| **Framework** | Streamlit |
| **Language Model Integration** | LangChain |
| **Data Storage** | SQLite |
| **Environment Management** | python-dotenv |

### Workflow Design

The application implements a conditional state machine with the following flow:

```
User Input
    ↓
[validate_query] - Check if recipe-related using LLM
    ↓
   ╱─ is_recipe_query=True → [generate_recipe] → END
  ╱
 ╱─ is_recipe_query=False → [non_recipe] → END
```

### State Management

```python
class RecipeState(TypedDict):
    user_input: str           # User's message
    is_recipe_query: bool     # Classification result
    recipe_response: str      # Generated recipe or response
    error_message: str        # Error handling
```

## 📋 Core Components

### 1. **Query Validation** (`validate_recipe_query`)
- Uses LLM to determine if user input is recipe-related
- Handles edge cases and non-recipe queries gracefully
- Implements error handling for API failures

### 2. **Recipe Generation** (`generate_recipe_response`)
- Generates detailed, professional recipes
- Includes ingredients, instructions, tips, and serving suggestions
- Provides expert-level culinary guidance

### 3. **Non-Recipe Handler** (`handle_non_recipe_query`)
- Politely redirects users for non-recipe queries
- Maintains context and user engagement

### 4. **Session Management**
- **`initialize_session_state()`**: Sets up chat storage and session variables
- **`create_session()`**: Generates new chat sessions with unique IDs
- **`delete_message()`**: Removes messages and updates database
- **`append_message()`**: Persists messages to SQLite

### 5. **UI/UX Components**
- **`animate_title()`**: Animated typewriter effect for app title
- **`md_to_html()`**: Markdown to HTML conversion for formatted recipes
- **`render_styles()`**: Custom CSS for polished visual design

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API Key
- SQLite (bundled with Python)

### Installation

```bash
# Clone the repository
git clone <your-repo>
cd <project-directory>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application

```bash
streamlit run Recepe_chat.py
```

The application will open in your default browser at `http://localhost:8501`

## 💡 Usage Examples

### Example 1: Recipe Query
```
User: "How to make Beef Karahi?"
Bot: [Provides detailed recipe with ingredients, steps, and tips]
```

### Example 2: Non-Recipe Query
```
User: "What's the weather today?"
Bot: "Sorry, I can only provide information about recipes and cooking. Please ask me about how to make a specific dish!"
```

### Example 3: Multi-Turn Conversation
```
User: "How to make chicken biryani?"
Bot: [Detailed biryani recipe]

User: "Can I use yogurt instead of cream?"
Bot: [Provides variation and substitution guidance]
```

## 📁 Project Structure

```
.
├── Recepe_chat.py                    # Main application file
├── recipe_chatbot.sqlite             # SQLite database
├── recipe_chatbot.sqlite-shm         # SQLite shared memory file
├── recipe_chatbot.sqlite-wal         # SQLite write-ahead log
├── .env                              # Environment variables
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## 🔧 Database Schema

### `chat_messages` Table

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | TEXT | Unique session identifier |
| `message_index` | INTEGER | Message order within session |
| `role` | TEXT | "user" or "assistant" |
| `content` | TEXT | Message text |
| PRIMARY KEY | (session_id, message_index) | Composite key |

## 🎨 UI Features

### Custom Styling
- **Dark theme** with warm, culinary-inspired colors
- **Playfair Display font** for elegant headings
- **Inter font** for readable body text
- **Badge design** with emoji and gradient backgrounds
- **Responsive layout** for desktop and mobile

### Interactive Elements
- **Animated title** on first load
- **Chat message bubbles** with role-based styling
- **Delete message** functionality within conversations
- **New chat** button for creating fresh sessions
- **Session management** sidebar with previous conversations

## 🚨 Error Handling

The application includes comprehensive error handling:

```python
try:
    response = model.invoke(messages)
    state["error_message"] = ""
except Exception as exc:
    state["error_message"] = f"Error: {exc}"
```

Failures in LLM calls don't crash the app; users receive graceful error messages.

## 🔐 Data Privacy

- All conversations are stored locally in SQLite
- No data is sent to external services except OpenAI API
- Users can delete individual messages or entire chat sessions
- Session data persists across app restarts

## 📊 Performance Considerations

- **SQLite Checkpointer**: LangGraph uses SQLite for state persistence
- **Session State Caching**: In-memory session management for quick access
- **Message Indexing**: Efficient message lookup and updates
- **Lazy Loading**: Chat sessions loaded only when needed

## 🎓 Learning Outcomes

This project demonstrates:

1. **LangGraph Expertise**
   - Building conditional state machines
   - Implementing checkpointer for persistence
   - Multi-node workflows with routing

2. **LangChain Integration**
   - Chat message handling (SystemMessage, HumanMessage)
   - LLM invocation and response processing
   - Error handling in LLM chains

3. **Streamlit Development**
   - Session state management
   - Custom CSS styling
   - Interactive UI components
   - Message history management

4. **Database Design**
   - SQLite schema design
   - CRUD operations
   - Data persistence patterns

5. **Software Engineering**
   - Type hints and type safety
   - Error handling and graceful degradation
   - Code organization and modularity
   - Environment configuration

## 🚀 Future Enhancements

- [ ] Add recipe difficulty levels and cooking time estimates
- [ ] Implement user ratings and recipe reviews
- [ ] Add dietary restrictions filtering (vegetarian, vegan, gluten-free)
- [ ] Integrate with recipe APIs for more comprehensive data
- [ ] Add image generation for recipes using DALL-E
- [ ] Implement recipe scaling based on serving size
- [ ] Add nutritional information parsing
- [ ] Multi-language support
- [ ] Export recipes to PDF
- [ ] Voice input/output capability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as a portfolio project demonstrating expertise in:
- AI/LLM application development
- State machine workflows
- Modern Python web frameworks
- Database design and management

## 📧 Contact

For questions or inquiries about this project, feel free to reach out.

---

**Last Updated**: March 2026


"""
Steam Game Recommendation System - Streamlit Application
Production-ready version with proper error handling and optimizations
"""
import streamlit as st
import pandas as pd
from typing import List
import logging
from config import config
from data_loader import DataLoader
from recommendation_engine import RecommendationEngine, Recommendation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state="expanded",
)


# Custom CSS for better UI
st.markdown("""
<style>
    .game-card {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .similarity-score {
        color: #666;
        font-size: 0.9em;
    }
    .stAlert {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with contact information and statistics"""
    with st.sidebar:
        st.title("📞 Contact Information")
        st.divider()
        
        contact = config.CONTACT_INFO
        st.markdown(f"[📸 Instagram]({contact['instagram']})")
        st.markdown(f"[💼 LinkedIn]({contact['linkedin']})")
        st.markdown(f"📧 Email: {contact['email']}")
        
        st.divider()
        
        # Add statistics if engine is available
        if 'engine' in st.session_state:
            st.title("📊 Statistics")
            stats = st.session_state.engine.get_statistics()
            st.metric("Total Games", stats['total_games'])
            st.metric("Games with Images", stats['games_with_images'])
            st.metric("Coverage", f"{stats['coverage']:.1f}%")


def render_recommendation_card(rec: Recommendation, col):
    """
    Render a single recommendation card
    
    Args:
        rec: Recommendation object
        col: Streamlit column to render in
    """
    with col:
        # Display image if available
        if rec.image_url:
            st.image(
                rec.image_url,
                use_column_width=True,
                caption=f"Similarity: {rec.similarity_score:.2%}"
            )
        else:
            st.info("🎮 No image available")
        
        # Game name
        st.subheader(rec.game_name)
        
        # Similarity score with styling
        st.markdown(
            f'<p class="similarity-score">Match: {rec.similarity_score:.1%}</p>',
            unsafe_allow_html=True
        )


def render_recommendations(recommendations: List[Recommendation]):
    """
    Render all recommendations in a grid layout
    
    Args:
        recommendations: List of Recommendation objects
    """
    st.header("🎯 Recommended Games")
    st.markdown("---")
    
    # Create 3-column grid
    n_cols = 3
    rows = [recommendations[i:i+n_cols] for i in range(0, len(recommendations), n_cols)]
    
    for row in rows:
        cols = st.columns(n_cols)
        for rec, col in zip(row, cols):
            render_recommendation_card(rec, col)


def initialize_app():
    """Initialize the application and load all data"""
    if 'initialized' not in st.session_state:
        try:
            # Load all data
            game_data, similarity, image_data, game_index = DataLoader.load_all_data()
            
            # Initialize recommendation engine
            engine = RecommendationEngine(
                game_data=game_data,
                similarity_matrix=similarity,
                image_data=image_data,
                game_index=game_index
            )
            
            # Store in session state
            st.session_state.engine = engine
            st.session_state.game_names = sorted(game_index.keys())
            st.session_state.initialized = True
            
            logger.info("Application initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            st.error(f"Failed to initialize application: {e}")
            st.stop()


def main():
    """Main application logic"""
    
    # Initialize
    initialize_app()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.title("🎮 Steam Game Recommendation System")
    st.markdown(
        "Discover games similar to your favorites! "
        "This system uses content-based filtering to analyze game tags, "
        "descriptions, and genres."
    )
    
    # Info banner
    st.info("ℹ️ This database contains games released between 2021 and 2023")
    
    st.markdown("---")
    
    # Game selection
    st.subheader("Select a Game")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_game = st.selectbox(
            "Choose or search for a game:",
            options=[''] + st.session_state.game_names,
            index=0,
            help="Start typing to search for a game"
        )
    
    with col2:
        # Number of recommendations slider
        top_n = st.slider(
            "Recommendations",
            min_value=3,
            max_value=12,
            value=6,
            step=3,
            help="Number of games to recommend"
        )
    
    # Show selected game
    if selected_game:
        st.success(f"✅ Selected: **{selected_game}**")
        
        # Find recommendations button
        if st.button("🔍 Find Similar Games", type="primary", use_container_width=True):
            
            # Show loading state
            with st.spinner("🎲 Analyzing game similarities..."):
                try:
                    # Get recommendations
                    recommendations = st.session_state.engine.get_cached_recommendations(
                        selected_game,
                        top_n=top_n
                    )
                    
                    # Show success animation
                    st.balloons()
                    st.toast("✨ Recommendations found!", icon="🎮")
                    
                    # Render recommendations
                    render_recommendations(recommendations)
                    
                    # Success message
                    st.success(f"✨ Found {len(recommendations)} similar games!")
                    
                except ValueError as e:
                    # Game not found
                    st.error(str(e))
                    logger.warning(f"Game not found: {selected_game}")
                    
                    # Suggest random games
                    st.info("💡 Try one of these popular games:")
                    random_games = st.session_state.engine.get_random_games(5)
                    for game in random_games:
                        st.write(f"• {game}")
                
                except Exception as e:
                    # Unexpected error
                    st.error("An unexpected error occurred. Please try again.")
                    logger.error(f"Recommendation error: {e}")
    
    else:
        # Show placeholder when no game selected
        st.info("👆 Select a game from the dropdown above to get started!")
        
        # Show some random game suggestions
        with st.expander("💡 Need inspiration? Try these games:"):
            random_games = st.session_state.engine.get_random_games(10)
            cols = st.columns(2)
            for idx, game in enumerate(random_games):
                cols[idx % 2].write(f"• {game}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit | 
            Data from Steam (2021-2023) | 
            Content-based filtering with cosine similarity</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

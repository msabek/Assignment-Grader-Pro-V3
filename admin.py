import streamlit as st
from scoring_criteria import ScoringCriteria
import pandas as pd

def admin_page():
    st.title("🔧 Marking Scheme Configuration")
    
    # Initialize scoring criteria
    scoring_criteria = ScoringCriteria()
    current_criteria = scoring_criteria.get_criteria()
    
    # Weights section
    st.subheader("Score Weights")
    similarity_weight = st.slider(
        "Similarity Weight",
        min_value=0,
        max_value=20,
        value=current_criteria['similarity_weight'],
        help="Weight for text similarity score"
    )
    
    concept_weight = st.slider(
        "Concept Weight",
        min_value=0,
        max_value=20,
        value=current_criteria['concept_weight'],
        help="Weight for concept coverage"
    )
    
    concept_penalty = st.slider(
        "Concept Penalty",
        min_value=0.0,
        max_value=2.0,
        value=current_criteria['concept_penalty'],
        step=0.1,
        help="Points deducted per missing concept"
    )
    
    # Thresholds section
    st.subheader("Score Thresholds")
    excellent_threshold = st.slider(
        "Excellent Threshold",
        min_value=0.0,
        max_value=1.0,
        value=current_criteria['thresholds']['excellent'],
        step=0.05,
        help="Minimum similarity for excellent grade"
    )
    
    good_threshold = st.slider(
        "Good Threshold",
        min_value=0.0,
        max_value=excellent_threshold,
        value=current_criteria['thresholds']['good'],
        step=0.05,
        help="Minimum similarity for good grade"
    )
    
    satisfactory_threshold = st.slider(
        "Satisfactory Threshold",
        min_value=0.0,
        max_value=good_threshold,
        value=current_criteria['thresholds']['satisfactory'],
        step=0.05,
        help="Minimum similarity for satisfactory grade"
    )
    
    # Create new marking scheme dictionary
    new_criteria = {
        'similarity_weight': similarity_weight,
        'concept_weight': concept_weight,
        'concept_penalty': concept_penalty,
        'thresholds': {
            'excellent': excellent_threshold,
            'good': good_threshold,
            'satisfactory': satisfactory_threshold
        }
    }
    
    # Apply changes button
    if st.button("Apply Changes", type="primary"):
        scoring_criteria.update_criteria(new_criteria)
        st.success("Marking scheme updated successfully!")
    
    if st.button("Reset to Default"):
        scoring_criteria.reset_to_default()
        st.success("Reset to default marking scheme")
        st.rerun()
    
    # Display current scheme as JSON
    with st.expander("View Current Configuration"):
        st.json(new_criteria)

if __name__ == "__main__":
    admin_page()

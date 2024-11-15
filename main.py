import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import numpy as np
from file_processor import FileProcessor
from groq_comparator import GroqTextComparator
from ollama_comparator import OllamaTextComparator
from openai_comparator import OpenAIComparator
from openai_compatible_comparator import OpenAICompatibleComparator
from mistral_comparator import MistralComparator
from scoring import ScoreCalculator
from report_generator import ReportGenerator
from statistical_analyzer import StatisticalAnalyzer
import io
import zipfile
import os


def show_evaluation_settings(groq_comparator, ollama_comparator,
                           openai_compatible_comparator, openai_comparator,
                           mistral_comparator):
    """Configure evaluation settings including LLM provider and marking criteria."""
    st.sidebar.markdown("### Evaluation Settings")

    # Initialize session state for model selection and API keys
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = groq_comparator.available_models[0]
    if 'selected_ollama_model' not in st.session_state:
        st.session_state.selected_ollama_model = ollama_comparator.available_models[0]
    if 'selected_openai_model' not in st.session_state:
        st.session_state.selected_openai_model = openai_compatible_comparator.available_models[0]
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'mistral_api_key' not in st.session_state:
        st.session_state.mistral_api_key = ""
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = os.environ.get('GROQ_API_KEY', '')
    if 'selected_cloud_openai_model' not in st.session_state:
        st.session_state.selected_cloud_openai_model = openai_comparator.available_models[0]
    if 'selected_mistral_model' not in st.session_state:
        st.session_state.selected_mistral_model = mistral_comparator.available_models[0]
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.5
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 4000
    if 'openai_models' not in st.session_state:
        st.session_state.openai_models = openai_comparator.available_models
    if 'mistral_models' not in st.session_state:
        st.session_state.mistral_models = mistral_comparator.available_models
    if 'groq_models' not in st.session_state:
        st.session_state.groq_models = groq_comparator.available_models
    if 'api_status' not in st.session_state:
        st.session_state.api_status = {}
    if 'stats_data' not in st.session_state:
        st.session_state.stats_data = {
            'boxplot': go.Figure(),  # Initialize empty figure
            'distribution': go.Figure(),
            'performance': go.Figure(),
            'bands': {}
        }

    # LLM Provider Selection
    llm_provider = st.sidebar.radio(
        "Select LLM Provider", [
            "Groq (Cloud)", "OpenAI (Cloud)", "Mistral (Cloud)", "Ollama (Local)",
            "OpenAI-compatible Server"
        ],
        help="Choose the AI model provider for evaluation")

    # Function to handle API key validation and model fetching
    def validate_api_key(provider, api_key, comparator, models_key):
        status_placeholder = st.sidebar.empty()

        # Show loading state
        with status_placeholder:
            with st.spinner(f'Validating {provider} API key...'):
                try:
                    # Special handling for Groq which uses environment variables
                    if provider == 'Groq':
                        available_models = comparator.available_models
                        if available_models:
                            st.session_state[models_key] = available_models
                            st.session_state.api_status[provider] = {
                                'valid': True,
                                'message': f"✅ {provider} API key is valid"
                            }
                            return True
                        else:
                            st.session_state.api_status[provider] = {
                                'valid': False,
                                'message': "❌ Invalid or missing Groq API key"
                            }
                            return False
                    else:
                        # For other providers that use set_api_key
                        available_models = comparator.set_api_key(api_key)
                        st.session_state[models_key] = available_models
                        st.session_state.api_status[provider] = {
                            'valid': True,
                            'message': f"✅ {provider} API key set successfully"
                        }
                        return True
                except ValueError as e:
                    st.session_state.api_status[provider] = {
                        'valid': False,
                        'message': f"❌ {str(e)}"
                    }
                    # Reset to default models on error
                    st.session_state[models_key] = comparator.available_models
                    return False
                except Exception as e:
                    st.session_state.api_status[provider] = {
                        'valid': False,
                        'message': f"❌ Unexpected error: {str(e)}"
                    }
                    st.session_state[models_key] = comparator.available_models
                    return False

    # Model Selection based on provider
    if llm_provider == "Groq (Cloud)":
        # Display current API key status for Groq
        groq_status = st.sidebar.empty()
        if 'Groq' in st.session_state.api_status:
            groq_status.markdown(st.session_state.api_status['Groq']['message'])
        else:
            # Initial validation of environment variable
            is_valid = validate_api_key('Groq', None, groq_comparator, 'groq_models')
            if is_valid:
                groq_status.success("✅ Groq API key is valid")
            else:
                groq_status.warning("⚠️ Groq API key is not set in environment variables")

        # Model selection dropdown
        st.sidebar.selectbox("Select Groq Model",
                           st.session_state.groq_models,
                           key='selected_model',
                           help="Choose a Groq model for evaluation")

    elif llm_provider == "OpenAI (Cloud)":
        # API Key input with clear button
        api_key_col, clear_col = st.sidebar.columns([4, 1])
        with api_key_col:
            api_key = st.text_input("OpenAI API Key",
                                  type="password",
                                  value=st.session_state.openai_api_key,
                                  key="openai_api_key",
                                  help="Enter your OpenAI API key")
        with clear_col:
            if st.button("Clear", key="clear_openai"):
                st.session_state.openai_api_key = ""
                st.session_state.openai_models = openai_comparator.available_models
                if 'api_status' in st.session_state:
                    st.session_state.api_status.pop('OpenAI', None)
                st.rerun()

        # Update OpenAI comparator with API key and fetch models
        if api_key:
            is_valid = validate_api_key('OpenAI', api_key, openai_comparator,
                                      'openai_models')

            # Show status message
            if 'OpenAI' in st.session_state.api_status:
                st.sidebar.markdown(st.session_state.api_status['OpenAI']['message'])

            # Update model selection if needed
            if is_valid and st.session_state.selected_cloud_openai_model not in st.session_state.openai_models:
                st.session_state.selected_cloud_openai_model = st.session_state.openai_models[
                    0]

        # Model selection dropdown
        st.sidebar.selectbox("Select OpenAI Model",
                           st.session_state.openai_models,
                           key='selected_cloud_openai_model',
                           help="Choose an OpenAI model for evaluation")

    elif llm_provider == "Mistral (Cloud)":
        # API Key input with clear button
        api_key_col, clear_col = st.sidebar.columns([4, 1])
        with api_key_col:
            api_key = st.text_input("Mistral API Key",
                                  type="password",
                                  value=st.session_state.mistral_api_key,
                                  key="mistral_api_key",
                                  help="Enter your Mistral API key")
        with clear_col:
            if st.button("Clear", key="clear_mistral"):
                st.session_state.mistral_api_key = ""
                st.session_state.mistral_models = mistral_comparator.available_models
                if 'api_status' in st.session_state:
                    st.session_state.api_status.pop('Mistral', None)
                st.rerun()

        # Update Mistral comparator with API key and fetch models
        if api_key:
            is_valid = validate_api_key('Mistral', api_key, mistral_comparator,
                                      'mistral_models')

            # Show status message
            if 'Mistral' in st.session_state.api_status:
                st.sidebar.markdown(st.session_state.api_status['Mistral']['message'])

            # Update model selection if needed
            if is_valid and st.session_state.selected_mistral_model not in st.session_state.mistral_models:
                st.session_state.selected_mistral_model = st.session_state.mistral_models[
                    0]

        # Model selection dropdown
        st.sidebar.selectbox("Select Mistral Model",
                           st.session_state.mistral_models,
                           key='selected_mistral_model',
                           help="Choose a Mistral model for evaluation")

    elif llm_provider == "Ollama (Local)":
        # Check Ollama connection status
        ollama_status = st.sidebar.empty()
        try:
            if not ollama_comparator._check_ollama_connection():
                ollama_status.warning(
                    "⚠️ Ollama service not running. Please install and start Ollama.")
            else:
                ollama_status.success("✅ Ollama service running")
        except Exception as e:
            ollama_status.warning(f"⚠️ Ollama service not running: {str(e)}")

        st.sidebar.selectbox("Select Ollama Model",
                           ollama_comparator.available_models,
                           key='selected_ollama_model',
                           help="Choose an Ollama model for evaluation")
    else:  # OpenAI-compatible Server
        # Check server connection status
        server_status = st.sidebar.empty()
        connection_status = openai_compatible_comparator.get_connection_status()
        if connection_status['available']:
            server_status.success("✅ OpenAI-compatible server running")
        else:
            server_status.warning(f"⚠️ {connection_status['error']}")

        st.sidebar.selectbox("Select Model",
                           openai_compatible_comparator.available_models,
                           key='selected_openai_model',
                           help="Choose a model from the OpenAI-compatible server")

    # Add model parameters controls
    st.sidebar.markdown("### Model Parameters")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Controls randomness in the model's output. Lower values make the output more focused and deterministic."
    )
    max_tokens = st.sidebar.number_input(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=st.session_state.max_tokens,
        step=100,
        help="Maximum number of tokens in the model's response.")

    # Update session state
    st.session_state.temperature = temperature
    st.session_state.max_tokens = max_tokens

    # Marking Criteria
    st.sidebar.markdown("### Marking Criteria")
    marking_criteria = st.sidebar.text_area(
        "Enter Marking Criteria",
        value="Based on the [Student Assignments], [Ideal Solutions], grade each student's assignment, and provide the final score out of 20 and the gained mark for each question.",
        help="Enter specific marking criteria for evaluation")

    return llm_provider, marking_criteria, temperature, max_tokens


def display_feedback(feedback, score, unique_id):
    """Display formatted feedback with score."""
    st.markdown(f"**Score:** {score:.1f}/20")
    st.text_area("Detailed Feedback",
                 value=feedback,
                 height=200,
                 disabled=True,
                 key=f"feedback_{unique_id}")


def generate_sample_results():
    """Generate sample results for testing the Statistical Analysis tab."""
    import numpy as np

    num_samples = 30  # Increased sample size for better distribution
    results = []

    # Generate random scores with a more realistic distribution
    # Using a truncated normal distribution centered around 14 (70%)
    scores = np.random.normal(14, 3, num_samples)
    scores = np.clip(scores, 0, 20)  # Ensure scores are within valid range

    for i, score in enumerate(scores, 1):
        results.append({
            'filename': f'student_{i}.pdf',
            'student_id': f'{1000 + i}',
            'score': float(score),
            'feedback': f'Sample feedback for student {i}\nQuestion 1: [Score: {score/3:.1f}/6.67] Good understanding shown.\nQuestion 2: [Score: {score/3:.1f}/6.67] Clear explanation provided.\nQuestion 3: [Score: {score/3:.1f}/6.67] Well-structured response.'
        })

    return results


def main():
    st.set_page_config(page_title="Assignment Grader Pro",
                       page_icon="📚",
                       layout="wide")

    # Initialize session state for results and stats
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
    if 'stats_data' not in st.session_state:
        st.session_state.stats_data = {
            'boxplot': go.Figure(),  # Initialize empty figure
            'distribution': go.Figure(),
            'performance': go.Figure(),
            'bands': {},
            'last_update': None
        }

    # Add logo and title in a horizontal layout
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("university-of-alberta-logo.png", width=200)
    with col2:
        st.title("📚 Assignment Grader Pro")

    # Initialize components
    file_processor = FileProcessor()
    groq_comparator = GroqTextComparator()
    ollama_comparator = OllamaTextComparator()
    openai_compatible_comparator = OpenAICompatibleComparator()
    openai_comparator = OpenAIComparator()
    mistral_comparator = MistralComparator()
    score_calculator = ScoreCalculator()
    report_generator = ReportGenerator()
    statistical_analyzer = StatisticalAnalyzer()

    # Show evaluation settings
    llm_provider, marking_criteria, temperature, max_tokens = show_evaluation_settings(
        groq_comparator, ollama_comparator, openai_compatible_comparator,
        openai_comparator, mistral_comparator)

    # Create tabs
    upload_tab, results_tab, stats_tab, reports_tab = st.tabs(
        ['📄 Upload Files', '📊 Results', '📈 Statistical Analysis', '📑 Reports'])

    # Upload Files tab
    with upload_tab:
        st.markdown("### Upload Assignment Files")
        col1, col2 = st.columns(2)

        with col1:
            solution_file = st.file_uploader("Solution PDF",
                                           type=['pdf'],
                                           help="Upload the solution/marking guide PDF")

        with col2:
            student_files = st.file_uploader(
                "Student Submissions",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload student PDFs (format: StudentID_Assignment.pdf)")

        if solution_file and student_files:
            if st.button("Start Evaluation", type="primary", use_container_width=True):
                try:
                    with st.spinner():
                        # Process solution file
                        solution_text = file_processor.extract_text(solution_file)

                        # Process and evaluate student files
                        results = []
                        progress_container = st.container()

                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            time_text = st.empty()

                            total_files = len(student_files)
                            start_time = time.time()

                            for idx, student_file in enumerate(student_files, 1):
                                try:
                                    # Update progress
                                    progress = idx / total_files
                                    progress_bar.progress(progress)
                                    status_text.write(
                                        f"Processing {student_file.name} ({idx}/{total_files})")

                                    # Process file
                                    student_text = file_processor.extract_text(student_file)
                                    student_id = file_processor.extract_student_id(
                                        student_file.name)

                                    # Compare texts using selected provider
                                    comparison_result = None
                                    if llm_provider == "Groq (Cloud)":
                                        # No need to validate API key, it's done in settings
                                        comparison_result = groq_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
                                    elif llm_provider == "OpenAI (Cloud)":
                                        if not st.session_state.openai_api_key:
                                            st.error(
                                                "Please provide your OpenAI API key in the settings")
                                            break
                                        comparison_result = openai_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_cloud_openai_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
                                    elif llm_provider == "Mistral (Cloud)":
                                        if not st.session_state.mistral_api_key:
                                            st.error(
                                                "Please provide your Mistral API key in the settings")
                                            break
                                        comparison_result = mistral_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_mistral_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
                                    elif llm_provider == "Ollama (Local)":
                                        comparison_result = ollama_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_ollama_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            max_tokens=max_tokens)
                                    else:  # OpenAI-compatible Server
                                        comparison_result = openai_compatible_comparator.compare_texts(
                                            student_text=student_text,
                                            solution_text=solution_text,
                                            model_name=st.session_state.selected_openai_model,
                                            marking_criteria=marking_criteria,
                                            temperature=temperature,
                                            max_tokens=max_tokens)

                                    if comparison_result:
                                        feedback = comparison_result.get('detailed_feedback',
                                                                    'No feedback provided')
                                        score = score_calculator.extract_score(feedback)
                                        enhanced_feedback = score_calculator.generate_feedback(
                                            feedback, score)

                                        results.append({
                                            'filename': student_file.name,
                                            'student_id': student_id,
                                            'score': score,
                                            'feedback': enhanced_feedback
                                        })

                                        # Update DataFrame
                                        st.session_state.results_df = pd.DataFrame(results)

                                except Exception as e:
                                    st.error(
                                        f"Error processing {student_file.name}: {str(e)}")

                            # Update progress and timing
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            time_text.write(
                                f"Total processing time: {elapsed_time:.2f} seconds")

                        # Store results in session state
                        st.session_state.results = results

                        # Show completion message
                        st.success(
                            f"✅ Evaluation completed for {len(results)} submissions")

                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")

    # Results tab
    with results_tab:
        st.markdown("### All Scores")
        if st.session_state.results_df is None or st.session_state.results_df.empty:
            st.info("No results available yet. Please process some submissions first.")
        else:
            # Create and display the scores table
            scores_df = st.session_state.results_df[['student_id', 'filename', 'score']].sort_values('score', ascending=False)
            st.dataframe(
                scores_df,
                column_config={
                    'student_id': 'Student ID',
                    'filename': 'Filename',
                    'score': st.column_config.NumberColumn('Score (/20)', format="%.1f")
                },
                hide_index=True
            )

            st.markdown("### Detailed Results")
            for result in st.session_state.results:
                with st.expander(f"📄 {result['filename']} (Student ID: {result['student_id']})"):
                    display_feedback(result['feedback'], result['score'],
                                  f"result_{result['student_id']}")

    # Statistical Analysis tab
    with stats_tab:
        st.markdown("### Statistical Analysis")
        if st.session_state.results:
            df = pd.DataFrame(st.session_state.results)

            # Update statistical visualizations
            st.session_state.stats_data = {
                'boxplot': statistical_analyzer.generate_score_boxplot(df['score'].tolist()),
                'distribution': statistical_analyzer.generate_score_distribution(df['score'].tolist())['plot'],
                'performance': statistical_analyzer.generate_performance_pie_chart(df['score'].tolist()),
                'bands': statistical_analyzer.generate_performance_bands(df['score'].tolist())
            }

            # Display visualizations
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(st.session_state.stats_data['distribution'], use_container_width=True)
                st.plotly_chart(st.session_state.stats_data['boxplot'], use_container_width=True)
            with col2:
                st.plotly_chart(st.session_state.stats_data['performance'], use_container_width=True)
                st.markdown("### Performance Bands")
                for band, count in st.session_state.stats_data['bands'].items():
                    st.write(f"{band}: {count} students")
        else:
            st.info("No data available for statistical analysis. Please evaluate some submissions first.")

    # Reports tab
    with reports_tab:
        st.markdown("### Generate Reports")
        if not st.session_state.results:
            st.info("No results available. Please evaluate files first.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Generate Complete Report", use_container_width=True):
                    with st.spinner("Generating complete report..."):
                        try:
                            report_pdf = report_generator.generate_complete_report(
                                st.session_state.results,
                                st.session_state.results_df,
                                marking_criteria,
                                statistical_analyzer
                            )
                            st.download_button(
                                "📥 Download Complete Report",
                                report_pdf,
                                "complete_report.pdf",
                                "application/pdf",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error generating complete report: {str(e)}")

            with col2:
                if st.button("Download All Individual Reports", use_container_width=True):
                    with st.spinner("Generating individual reports..."):
                        try:
                            # Create a ZIP file in memory
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for result in st.session_state.results:
                                    # Generate individual PDF report
                                    pdf_data = report_generator.generate_pdf_report(result)

                                    # Add PDF to ZIP with a meaningful filename
                                    filename = f"report_{result['student_id']}_{result['filename'].split('.')[0]}.pdf"
                                    zip_file.writestr(filename, pdf_data)

                            # Prepare ZIP file for download
                            zip_buffer.seek(0)
                            st.download_button(
                                "📥 Download All Individual Reports (ZIP)",
                                zip_buffer.getvalue(),
                                "individual_reports.zip",
                                "application/zip",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"Error generating individual reports: {str(e)}")

            # Individual report generation section
            st.markdown("### Generate Individual Reports")
            for result in st.session_state.results:
                with st.expander(f"📄 Generate report for {result['filename']} (Student ID: {result['student_id']})"):
                    try:
                        report_pdf = report_generator.generate_pdf_report(result)
                        st.download_button(
                            "📥 Download Individual Report",
                            report_pdf,
                            f"report_{result['student_id']}_{result['filename'].split('.')[0]}.pdf",
                            "application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error generating individual report: {str(e)}")
            # Add creator credits at the bottom of the page
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666666; padding: 20px;'>
        <p>Created by the University of Alberta, Department of Civil & Environmental Engineering. By: Mohamed sabek</p>
        <p>© 2024 University of Alberta. All rights reserved.</p>
        </div>
        """,
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
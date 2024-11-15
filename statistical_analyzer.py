import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import numpy as np
from scipy import stats
import warnings

class StatisticalAnalyzer:
    def __init__(self):
        """Initialize the StatisticalAnalyzer."""
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._min_score = 0
        self._max_score = 20
        self._colors = {
            'primary': '#2E86C1',
            'secondary': '#E74C3C',
            'success': '#2ECC71',
            'warning': '#F1C40F',
            'info': '#3498DB',
            'purple': '#9B59B6',
            'gray': '#95A5A6'
        }

    def _validate_scores(self, scores: List[float]) -> List[float]:
        """Validate and clean score data."""
        if not scores:
            return []
        
        valid_scores = []
        try:
            for score in scores:
                score_float = float(score)
                if self._min_score <= score_float <= self._max_score:
                    valid_scores.append(score_float)
                else:
                    print(f"Warning: Score {score_float} outside valid range [{self._min_score}, {self._max_score}]")
        except (ValueError, TypeError) as e:
            print(f"Error validating scores: {str(e)}")
        
        return valid_scores

    def generate_score_distribution(self, scores: List[float]) -> Dict[str, Any]:
        """Generate score distribution with enhanced visualization."""
        valid_scores = self._validate_scores(scores)
        stats_result = {
            'mean': 0,
            'median': 0,
            'std': 0,
            'q1': 0,
            'q3': 0,
            'count': 0,
            'skewness': 0,
            'kurtosis': 0
        }
        
        empty_result = {
            'plot': go.Figure(),
            'stats': stats_result
        }
        
        if not valid_scores:
            return empty_result
        
        try:
            df = pd.DataFrame(valid_scores, columns=['Score'])
            
            # Calculate statistics
            stats_result = {
                'mean': float(df['Score'].mean()),
                'median': float(df['Score'].median()),
                'std': float(df['Score'].std()),
                'q1': float(df['Score'].quantile(0.25)),
                'q3': float(df['Score'].quantile(0.75)),
                'count': len(df),
                'skewness': float(stats.skew(valid_scores)),
                'kurtosis': float(stats.kurtosis(valid_scores))
            }
            
            # Create histogram with optional KDE
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=valid_scores,
                nbinsx=min(len(valid_scores), 10),
                name='Score Distribution',
                marker_color=self._colors['primary'],
                opacity=0.7
            ))
            
            try:
                # Try to add KDE plot
                if len(set(valid_scores)) > 1:  # Only add KDE if we have different values
                    kde_x = np.linspace(min(valid_scores), max(valid_scores), 100)
                    kde = stats.gaussian_kde(valid_scores, bw_method='scott')
                    kde_y = kde(kde_x) * len(valid_scores) * (max(valid_scores) - min(valid_scores)) / 10
                    
                    fig.add_trace(go.Scatter(
                        x=kde_x,
                        y=kde_y,
                        name='Density',
                        line=dict(color=self._colors['secondary'], width=2),
                        mode='lines'
                    ))
            except Exception as kde_error:
                print(f"Note: KDE plot could not be generated: {str(kde_error)}")
            
            # Add mean and median lines
            mean_score = stats_result['mean']
            median_score = stats_result['median']
            
            fig.add_vline(x=mean_score, line_dash="dash", line_color=self._colors['success'],
                         annotation_text=f"Mean: {mean_score:.1f}")
            fig.add_vline(x=median_score, line_dash="dash", line_color=self._colors['purple'],
                         annotation_text=f"Median: {median_score:.1f}")
            
            fig.update_layout(
                title='Score Distribution',
                xaxis_title='Score (/20)',
                yaxis_title='Number of Students',
                template='plotly_white',
                showlegend=True,
                xaxis_range=[0, 20],
                bargap=0.1,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            return {
                'plot': fig,
                'stats': stats_result
            }
            
        except Exception as e:
            print(f"Error generating score distribution: {str(e)}")
            return empty_result

    def generate_score_boxplot(self, scores: List[float]) -> go.Figure:
        """Generate enhanced box plot with violin plot for score distribution."""
        valid_scores = self._validate_scores(scores)
        if not valid_scores:
            return go.Figure()
        
        try:
            fig = go.Figure()
            
            # Add violin plot
            fig.add_trace(go.Violin(
                y=valid_scores,
                box_visible=True,
                line_color=self._colors['primary'],
                fillcolor=self._colors['primary'],
                opacity=0.6,
                meanline_visible=True,
                points='all',
                jitter=0.05,
                pointpos=-0.1,
                marker=dict(
                    size=8,
                    opacity=0.7,
                    color=self._colors['info']
                ),
                name='Score Distribution'
            ))

            # Calculate statistics for annotations
            mean_score = np.mean(valid_scores)
            q1 = np.percentile(valid_scores, 25)
            median = np.median(valid_scores)
            q3 = np.percentile(valid_scores, 75)
            
            # Add annotations
            annotations = [
                dict(x=0, y=mean_score, text=f'Mean: {mean_score:.1f}',
                     showarrow=True, arrowhead=2, ax=50, ay=0,
                     font=dict(color=self._colors['success'])),
                dict(x=0, y=median, text=f'Median: {median:.1f}',
                     showarrow=True, arrowhead=2, ax=-50, ay=0,
                     font=dict(color=self._colors['purple'])),
                dict(x=0, y=q1, text=f'Q1: {q1:.1f}',
                     showarrow=True, arrowhead=2, ax=50, ay=0,
                     font=dict(color=self._colors['warning'])),
                dict(x=0, y=q3, text=f'Q3: {q3:.1f}',
                     showarrow=True, arrowhead=2, ax=-50, ay=0,
                     font=dict(color=self._colors['warning']))
            ]

            fig.update_layout(
                title='Score Distribution Violin Plot',
                yaxis_title='Score (/20)',
                yaxis_range=[0, 20],
                showlegend=False,
                template='plotly_white',
                annotations=annotations,
                violinmode='overlay'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error generating violin plot: {str(e)}")
            return go.Figure()

    def generate_performance_bands(self, scores: List[float]) -> Dict[str, int]:
        """Generate performance bands with enhanced granularity."""
        valid_scores = self._validate_scores(scores)
        empty_bands = {
            'Outstanding (18-20)': 0,
            'Excellent (16-17.9)': 0,
            'Very Good (14-15.9)': 0,
            'Good (12-13.9)': 0,
            'Satisfactory (10-11.9)': 0,
            'Pass (8-9.9)': 0,
            'Needs Improvement (0-7.9)': 0
        }
        
        if not valid_scores:
            return empty_bands
        
        try:
            df = pd.DataFrame(valid_scores, columns=['Score'])
            bands = {
                'Outstanding (18-20)': int(((df['Score'] >= 18) & (df['Score'] <= 20)).sum()),
                'Excellent (16-17.9)': int(((df['Score'] >= 16) & (df['Score'] < 18)).sum()),
                'Very Good (14-15.9)': int(((df['Score'] >= 14) & (df['Score'] < 16)).sum()),
                'Good (12-13.9)': int(((df['Score'] >= 12) & (df['Score'] < 14)).sum()),
                'Satisfactory (10-11.9)': int(((df['Score'] >= 10) & (df['Score'] < 12)).sum()),
                'Pass (8-9.9)': int(((df['Score'] >= 8) & (df['Score'] < 10)).sum()),
                'Needs Improvement (0-7.9)': int(((df['Score'] >= 0) & (df['Score'] < 8)).sum())
            }
            return bands
            
        except Exception as e:
            print(f"Error generating performance bands: {str(e)}")
            return empty_bands

    def generate_performance_pie_chart(self, scores: List[float]) -> go.Figure:
        """Generate enhanced pie chart for performance bands."""
        try:
            bands = self.generate_performance_bands(scores)
            
            # Only include non-zero bands
            non_zero_bands = {k: v for k, v in bands.items() if v > 0}
            
            if not non_zero_bands:
                return go.Figure()
            
            colors = [
                self._colors['success'],
                self._colors['info'],
                self._colors['purple'],
                self._colors['warning'],
                self._colors['secondary'],
                self._colors['primary'],
                self._colors['gray']
            ]
            
            fig = go.Figure(data=[go.Pie(
                labels=list(non_zero_bands.keys()),
                values=list(non_zero_bands.values()),
                hole=0.3,
                marker_colors=colors[:len(non_zero_bands)],
                textinfo='label+percent',
                textposition='outside',
                pull=[0.1 if v == max(non_zero_bands.values()) else 0 for v in non_zero_bands.values()]
            )])
            
            fig.update_layout(
                title='Performance Distribution',
                annotations=[dict(
                    text=f'Total: {sum(non_zero_bands.values())}',
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False
                )],
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            print(f"Error generating performance pie chart: {str(e)}")
            return go.Figure()

    def identify_outliers(self, scores: List[float]) -> Dict[str, Any]:
        """Identify statistical outliers with enhanced analysis."""
        valid_scores = self._validate_scores(scores)
        empty_result = {'low': [], 'high': [], 'stats': {}}
        
        if not valid_scores or len(valid_scores) < 4:
            return empty_result
        
        try:
            df = pd.DataFrame(valid_scores, columns=['Score'])
            
            q1 = df['Score'].quantile(0.25)
            q3 = df['Score'].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = max(0, q1 - 1.5 * iqr)
            upper_bound = min(20, q3 + 1.5 * iqr)
            
            outliers = {
                'low': df[df['Score'] < lower_bound]['Score'].tolist(),
                'high': df[df['Score'] > upper_bound]['Score'].tolist(),
                'stats': {
                    'q1': float(q1),
                    'q3': float(q3),
                    'iqr': float(iqr),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'total_outliers': len(df[df['Score'] < lower_bound]) + len(df[df['Score'] > upper_bound])
                }
            }
            
            return outliers
            
        except Exception as e:
            print(f"Error identifying outliers: {str(e)}")
            return empty_result

    def calculate_balance_index(self, scores: List[float]) -> Dict[str, float]:
        """Calculate enhanced balance index with additional metrics."""
        valid_scores = self._validate_scores(scores)
        empty_result = {'balance_index': 0.0, 'symmetry_score': 0.0, 'distribution_quality': 0.0}
        
        if not valid_scores or len(valid_scores) < 2:
            return empty_result
        
        try:
            # Calculate quartiles and other statistics
            q1 = np.percentile(valid_scores, 25)
            q3 = np.percentile(valid_scores, 75)
            median = np.median(valid_scores)
            mean = np.mean(valid_scores)
            
            # Calculate balance index
            balance = 1 - (abs((q3 - median) - (median - q1)) / (q3 - q1) if q3 != q1 else 0)
            
            # Calculate symmetry score (how close mean is to median)
            symmetry = 1 - min(abs(mean - median) / self._max_score, 1)
            
            # Calculate distribution quality (based on skewness)
            skewness = stats.skew(valid_scores)
            distribution_quality = 1 / (1 + abs(skewness))
            
            return {
                'balance_index': round(max(0, min(1, balance)), 3),
                'symmetry_score': round(symmetry, 3),
                'distribution_quality': round(distribution_quality, 3)
            }
            
        except Exception as e:
            print(f"Error calculating balance metrics: {str(e)}")
            return empty_result

    def calculate_discrimination_index(self, scores: List[float]) -> Dict[str, float]:
        """Calculate enhanced discrimination index with additional metrics."""
        valid_scores = self._validate_scores(scores)
        empty_result = {
            'discrimination_index': 0.0,
            'point_biserial': 0.0,
            'effectiveness': 0.0
        }
        
        if not valid_scores or len(valid_scores) < 4:
            return empty_result
        
        try:
            # Split into high and low groups (top 27% and bottom 27%)
            n = len(valid_scores)
            k = int(n * 0.27)
            sorted_scores = sorted(valid_scores)
            
            high_group = sorted_scores[-k:]
            low_group = sorted_scores[:k]
            
            # Calculate mean and standard deviation
            mean_high = np.mean(high_group)
            mean_low = np.mean(low_group)
            std_all = np.std(valid_scores)
            
            if std_all == 0:
                return empty_result
                
            p = len(high_group) / len(valid_scores)
            q = 1 - p
            
            # Calculate discrimination index
            discrimination = ((mean_high - mean_low) / std_all) * np.sqrt(p * q)
            
            # Calculate point-biserial correlation
            point_biserial = stats.pointbiserialr(
                [1] * len(high_group) + [0] * len(low_group),
                high_group + low_group
            )[0]
            
            # Calculate effectiveness (how well it separates high and low performers)
            effectiveness = (mean_high - mean_low) / self._max_score
            
            return {
                'discrimination_index': round(max(-1, min(1, discrimination)), 3),
                'point_biserial': round(float(point_biserial) if not np.isnan(point_biserial) else 0.0, 3),
                'effectiveness': round(max(0, min(1, effectiveness)), 3)
            }
            
        except Exception as e:
            print(f"Error calculating discrimination metrics: {str(e)}")
            return empty_result

    def analyze_section_performance(self, scores: List[float], section_max_points: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Analyze performance across different sections with enhanced metrics."""
        if not scores:
            return {}
        
        try:
            section_stats = {}
            for section, max_points in section_max_points.items():
                # Calculate normalized scores for the section
                section_scores = [score * (max_points/20) for score in scores]
                
                if not section_scores:
                    continue
                
                # Calculate basic statistics
                mean_score = np.mean(section_scores)
                median_score = np.median(section_scores)
                std_score = np.std(section_scores)
                
                # Calculate completion rate and difficulty index
                completion_rate = np.mean([score/max_points for score in section_scores])
                difficulty_index = 1 - completion_rate
                
                # Calculate discrimination power
                sorted_scores = sorted(section_scores)
                high_group = sorted_scores[-int(len(scores)*0.27):]
                low_group = sorted_scores[:int(len(scores)*0.27)]
                discrimination_power = (np.mean(high_group) - np.mean(low_group)) / max_points
                
                stats = {
                    'mean': round(mean_score, 2),
                    'median': round(median_score, 2),
                    'std': round(std_score, 2),
                    'max': round(max(section_scores), 2),
                    'min': round(min(section_scores), 2),
                    'completion_rate': round(completion_rate, 3),
                    'difficulty_index': round(difficulty_index, 3),
                    'discrimination_power': round(discrimination_power, 3)
                }
                section_stats[section] = stats
            
            return section_stats
            
        except Exception as e:
            print(f"Error analyzing section performance: {str(e)}")
            return {}

    def generate_effectiveness_report(self, scores: List[float], section_max_points: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive effectiveness report with enhanced metrics."""
        try:
            # Initialize report with default values
            report = {
                'overall_statistics': {
                    'balance_metrics': {
                        'balance_index': 0.0,
                        'symmetry_score': 0.0,
                        'distribution_quality': 0.0
                    },
                    'discrimination_metrics': {
                        'discrimination_index': 0.0,
                        'point_biserial': 0.0,
                        'effectiveness': 0.0
                    },
                    'reliability_coefficient': 0.0,
                    'score_distribution': {},
                    'outliers': {}
                },
                'section_performance': {},
                'recommendations': []
            }

            # Calculate effectiveness metrics only if we have valid scores
            valid_scores = self._validate_scores(scores)
            if valid_scores and len(valid_scores) > 1:
                # Add score distribution statistics
                distribution_data = self.generate_score_distribution(valid_scores)
                report['overall_statistics']['score_distribution'] = distribution_data['stats']
                
                # Add balance metrics
                report['overall_statistics']['balance_metrics'] = self.calculate_balance_index(valid_scores)
                
                # Add discrimination metrics
                report['overall_statistics']['discrimination_metrics'] = self.calculate_discrimination_index(valid_scores)
                
                # Add outlier analysis
                report['overall_statistics']['outliers'] = self.identify_outliers(valid_scores)
                
                # Calculate reliability coefficient
                report['overall_statistics']['reliability_coefficient'] = round(
                    np.corrcoef(valid_scores, valid_scores)[0, 1], 3
                )
                
                # Add section performance if available
                section_stats = self.analyze_section_performance(valid_scores, section_max_points)
                if section_stats:
                    report['section_performance'] = section_stats
                
                # Generate recommendations based on metrics
                recommendations = []
                
                # Check balance index
                balance_metrics = report['overall_statistics']['balance_metrics']
                if balance_metrics['balance_index'] < 0.7:
                    recommendations.append(
                        "Consider reviewing the marking scheme balance as the distribution shows significant skew."
                    )
                
                # Check discrimination power
                discrimination_metrics = report['overall_statistics']['discrimination_metrics']
                if discrimination_metrics['discrimination_index'] < 0.3:
                    recommendations.append(
                        "The assessment may need more discriminating questions to better differentiate student abilities."
                    )
                
                # Check outliers
                outliers = report['overall_statistics']['outliers']
                if outliers.get('stats', {}).get('total_outliers', 0) > len(valid_scores) * 0.1:
                    recommendations.append(
                        "High number of outlier scores detected. Consider reviewing these cases individually."
                    )
                
                report['recommendations'] = recommendations
            
            return report
            
        except Exception as e:
            print(f"Error generating effectiveness report: {str(e)}")
            return {
                'overall_statistics': {
                    'balance_metrics': {'balance_index': 0.0, 'symmetry_score': 0.0, 'distribution_quality': 0.0},
                    'discrimination_metrics': {'discrimination_index': 0.0, 'point_biserial': 0.0, 'effectiveness': 0.0},
                    'reliability_coefficient': 0.0,
                    'score_distribution': {},
                    'outliers': {}
                },
                'section_performance': {},
                'recommendations': []
            }

    def generate_effectiveness_plots(self, scores: List[float], section_max_points: Dict[str, float]) -> Dict[str, go.Figure]:
        """Generate enhanced visualizations for marking scheme effectiveness analysis."""
        try:
            plots = {}
            valid_scores = self._validate_scores(scores)
            
            # Generate section performance plot with enhanced visualization
            section_stats = self.analyze_section_performance(valid_scores, section_max_points)
            if section_stats:
                section_means = [stats['mean'] for stats in section_stats.values()]
                section_names = list(section_stats.keys())
                
                fig = go.Figure()
                
                # Add stacked bars for different metrics
                fig.add_trace(go.Bar(
                    name='Mean Score',
                    x=section_names,
                    y=section_means,
                    marker_color=self._colors['primary']
                ))
                
                # Add completion rates
                completion_rates = [stats['completion_rate'] * max_points 
                                 for stats, max_points in zip(section_stats.values(), section_max_points.values())]
                fig.add_trace(go.Bar(
                    name='Completion Rate',
                    x=section_names,
                    y=completion_rates,
                    marker_color=self._colors['success'],
                    opacity=0.7
                ))
                
                # Add discrimination power
                discrimination_power = [stats['discrimination_power'] * max_points 
                                     for stats, max_points in zip(section_stats.values(), section_max_points.values())]
                fig.add_trace(go.Scatter(
                    name='Discrimination Power',
                    x=section_names,
                    y=discrimination_power,
                    mode='lines+markers',
                    line=dict(color=self._colors['secondary'], width=2)
                ))
                
                fig.update_layout(
                    title='Section-wise Performance Analysis',
                    xaxis_title='Section',
                    yaxis_title='Score',
                    template='plotly_white',
                    barmode='group',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                plots['section_performance'] = fig
                
            # Generate enhanced score distribution plot
            if valid_scores:
                # Get effectiveness metrics
                effectiveness_report = self.generate_effectiveness_report(valid_scores, section_max_points)
                balance_metrics = effectiveness_report['overall_statistics']['balance_metrics']
                discrimination_metrics = effectiveness_report['overall_statistics']['discrimination_metrics']
                
                fig = go.Figure()
                
                # Add histogram with KDE
                fig.add_trace(go.Histogram(
                    x=valid_scores,
                    nbinsx=min(len(valid_scores), 10),
                    name='Score Distribution',
                    marker_color=self._colors['primary'],
                    opacity=0.7
                ))
                
                # Add KDE curve
                kde_x = np.linspace(min(valid_scores), max(valid_scores), 100)
                kde = stats.gaussian_kde(valid_scores)
                kde_y = kde(kde_x) * len(valid_scores) * (max(valid_scores) - min(valid_scores)) / 10
                
                fig.add_trace(go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    name='Density',
                    line=dict(color=self._colors['secondary'], width=2),
                    mode='lines'
                ))
                
                # Update layout with metrics
                fig.update_layout(
                    title=f'Score Distribution Analysis<br>' +
                          f'Balance Index: {balance_metrics["balance_index"]:.3f} | ' +
                          f'Discrimination Index: {discrimination_metrics["discrimination_index"]:.3f} | ' +
                          f'Effectiveness: {discrimination_metrics["effectiveness"]:.3f}',
                    xaxis_title='Score (/20)',
                    yaxis_title='Number of Students',
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                plots['score_distribution'] = fig
            
            return plots
            
        except Exception as e:
            print(f"Error generating effectiveness plots: {str(e)}")
            return {}
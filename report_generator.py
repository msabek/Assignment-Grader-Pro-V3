import pandas as pd
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
import gc

class ReportGenerator:
    def __init__(self):
        """Initialize the ReportGenerator with default styles."""
        self._styles = None
        self._buffer_size = 10 * 1024 * 1024  # 10MB default buffer size

    def _create_pdf_style(self):
        """Create and cache PDF styles."""
        if self._styles is not None:
            return self._styles
            
        try:
            styles = getSampleStyleSheet()
            styles.add(
                ParagraphStyle(
                    name='CustomTitle',
                    parent=styles['Title'],
                    fontSize=24,
                    spaceAfter=30,
                    textColor=colors.HexColor('#2E86C1')
                )
            )
            styles.add(
                ParagraphStyle(
                    name='CustomHeading',
                    parent=styles['Heading1'],
                    fontSize=16,
                    spaceAfter=12,
                    textColor=colors.HexColor('#2E86C1')
                )
            )
            styles.add(
                ParagraphStyle(
                    name='CustomBody',
                    parent=styles['Normal'],
                    fontSize=11,
                    spaceAfter=6
                )
            )
            styles.add(
                ParagraphStyle(
                    name='WarningText',
                    parent=styles['Normal'],
                    fontSize=11,
                    textColor=colors.HexColor('#E74C3C')
                )
            )
            
            self._styles = styles
            return styles
            
        except Exception as e:
            raise Exception(f"Error creating PDF styles: {str(e)}")

    def _add_logo(self, story):
        """Add University of Alberta logo to the report."""
        try:
            if os.path.exists('university-of-alberta-logo.png'):
                logo = Image('university-of-alberta-logo.png',
                            width=4*inch,
                            height=1*inch)

                logo_table = Table([[logo]], colWidths=[4.5*inch])
                logo_table.setStyle(
                    TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ])
                )

                story.append(logo_table)
                story.append(Spacer(1, 24))
            else:
                print("Warning: Logo file not found")
                
        except Exception as e:
            print(f"Warning: Could not add logo to report: {str(e)}")

    def _create_metrics_table(self, metrics, title):
        """Create a formatted table for metrics display."""
        try:
            data = [[Paragraph(title, self._create_pdf_style()['CustomHeading'])]]
            for key, value in metrics.items():
                if isinstance(value, dict):
                    # Handle nested metrics
                    for sub_key, sub_value in value.items():
                        formatted_key = sub_key.replace('_', ' ').title()
                        data.append([
                            f"{formatted_key}:",
                            f"{sub_value:.3f}" if isinstance(sub_value, float) else str(sub_value)
                        ])
                else:
                    formatted_key = key.replace('_', ' ').title()
                    data.append([
                        f"{formatted_key}:",
                        f"{value:.3f}" if isinstance(value, float) else str(value)
                    ])
            
            table = Table(data, colWidths=[200, 300])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8F9FA')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0'))
            ]))
            return table
        except Exception as e:
            print(f"Error creating metrics table: {str(e)}")
            return None

    def generate_complete_report(self, results, results_df, marking_criteria, statistical_analyzer):
        """Generate a comprehensive PDF report including all results and statistics."""
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=2 * cm,
                leftMargin=2 * cm,
                topMargin=2 * cm,
                bottomMargin=2 * cm
            )

            styles = self._create_pdf_style()
            story = []

            # Add logo and title
            self._add_logo(story)
            story.append(Paragraph("Comprehensive Assignment Report", styles['CustomTitle']))
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            story.append(Paragraph(f"Generated on: {timestamp}", styles['CustomBody']))
            story.append(Spacer(1, 20))

            # Calculate statistics
            valid_scores = results_df['score'].tolist()
            effectiveness_report = statistical_analyzer.generate_effectiveness_report(
                valid_scores, 
                {'total': 20.0}  # Default section for overall scores
            )

            # Add Statistical Analysis section
            story.append(Paragraph("Statistical Analysis", styles['CustomHeading']))
            
            # Basic Statistics
            basic_stats = {
                'Total Submissions': len(results),
                'Mean Score': f"{results_df['score'].mean():.2f}/20",
                'Median Score': f"{results_df['score'].median():.2f}/20",
                'Standard Deviation': f"{results_df['score'].std():.2f}",
                'Minimum Score': f"{results_df['score'].min():.2f}/20",
                'Maximum Score': f"{results_df['score'].max():.2f}/20"
            }
            story.append(self._create_metrics_table(basic_stats, "Basic Statistics"))
            story.append(Spacer(1, 20))

            # Balance and Discrimination Metrics
            balance_metrics = effectiveness_report['overall_statistics']['balance_metrics']
            discrimination_metrics = effectiveness_report['overall_statistics']['discrimination_metrics']
            
            story.append(self._create_metrics_table(
                {'Balance Metrics': balance_metrics},
                "Balance Analysis"
            ))
            story.append(Spacer(1, 20))
            
            story.append(self._create_metrics_table(
                {'Discrimination Metrics': discrimination_metrics},
                "Discrimination Analysis"
            ))
            story.append(Spacer(1, 20))

            # Recommendations
            if effectiveness_report['recommendations']:
                story.append(Paragraph("Recommendations", styles['CustomHeading']))
                for recommendation in effectiveness_report['recommendations']:
                    story.append(Paragraph(f"• {recommendation}", styles['CustomBody']))
                story.append(Spacer(1, 20))

            # Performance Bands
            story.append(Paragraph("Performance Distribution", styles['CustomHeading']))
            bands = statistical_analyzer.generate_performance_bands(valid_scores)
            bands_data = [[band, count] for band, count in bands.items()]
            bands_table = Table(bands_data, colWidths=[200, 200])
            bands_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0'))
            ]))
            story.append(bands_table)
            story.append(PageBreak())

            # Individual Results
            story.append(Paragraph("Individual Results", styles['CustomHeading']))
            results_data = [["Student ID", "Filename", "Score", "Status"]]
            
            mean_score = results_df['score'].mean()
            std_score = results_df['score'].std()
            
            for result in results:
                score = result['score']
                if score >= mean_score + std_score:
                    status = "Above Average"
                elif score <= mean_score - std_score:
                    status = "Below Average"
                else:
                    status = "Average"
                    
                results_data.append([
                    result['student_id'],
                    result['filename'],
                    f"{score:.1f}/20",
                    status
                ])

            results_table = Table(results_data, colWidths=[100, 200, 80, 100])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
            ]))
            story.append(results_table)
            story.append(Spacer(1, 20))
            story.append(PageBreak())

            # Individual Feedback Section
            story.append(Paragraph("Individual Feedback", styles['CustomHeading']))
            story.append(Spacer(1, 12))

            for result in results:
                # Student Info Header
                student_header = f"Student ID: {result['student_id']} - {result['filename']}"
                story.append(Paragraph(student_header, styles['CustomHeading']))
                
                # Score
                score_text = f"Score: {result['score']:.1f}/20"
                story.append(Paragraph(score_text, ParagraphStyle(
                    'ScoreStyle',
                    parent=styles['CustomBody'],
                    textColor=colors.HexColor('#2E86C1'),
                    fontSize=12,
                    spaceAfter=6
                )))
                
                # Detailed Feedback
                feedback_lines = result['feedback'].split('\n')
                for line in feedback_lines:
                    if line.strip():
                        # Check if line is a question header
                        if line.startswith(('Question', 'Total Score')):
                            story.append(Paragraph(line, ParagraphStyle(
                                'FeedbackHeader',
                                parent=styles['CustomBody'],
                                textColor=colors.HexColor('#34495E'),
                                fontSize=11,
                                spaceAfter=6,
                                spaceBefore=6,
                                fontName='Helvetica-Bold'
                            )))
                        else:
                            story.append(Paragraph(line, styles['CustomBody']))
                
                # Add spacing between students
                story.append(Spacer(1, 20))
                story.append(Table([['']], colWidths=[450], rowHeights=[1],
                    style=TableStyle([
                        ('LINEABOVE', (0,0), (-1,0), 1, colors.HexColor('#E0E0E0')),
                    ])
                ))
                story.append(Spacer(1, 20))

            # Build document and clean up
            doc.build(story)
            buffer.seek(0)
            
            # Force garbage collection to free memory
            story = None
            gc.collect()
            
            return buffer.getvalue()

        except Exception as e:
            raise Exception(f"Error generating complete PDF report: {str(e)}")

    def generate_pdf_report(self, result, marking_criteria=None):
        """Generate a PDF report for a single assignment."""
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=2 * cm,
                leftMargin=2 * cm,
                topMargin=2 * cm,
                bottomMargin=2 * cm
            )

            styles = self._create_pdf_style()
            story = []

            # Add logo and title
            self._add_logo(story)
            story.append(Paragraph("Assignment Feedback Report", styles['CustomTitle']))
            story.append(Spacer(1, 12))

            # Timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            story.append(Paragraph(f"Generated on: {timestamp}", styles['CustomBody']))
            story.append(Spacer(1, 20))

            # Student Details
            story.append(Paragraph("Assignment Details", styles['CustomHeading']))
            details_data = [
                ["Student ID:", result.get('student_id', 'unknown')],
                ["Filename:", result['filename']],
                ["Score:", f"{result['score']:.1f}/20"]
            ]
            
            details_table = Table(details_data, colWidths=[150, 350])
            details_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0'))
            ]))
            story.append(details_table)
            story.append(Spacer(1, 20))

            # Detailed Feedback
            story.append(Paragraph("AI-Generated Feedback", styles['CustomHeading']))
            feedback_lines = result['feedback'].split('\n')
            for line in feedback_lines:
                if line.strip():
                    story.append(Paragraph(line, styles['CustomBody']))
                    story.append(Spacer(1, 6))

            # Build document and clean up
            doc.build(story)
            buffer.seek(0)
            
            # Force garbage collection to free memory
            story = None
            gc.collect()
            
            return buffer.getvalue()
            
        except Exception as e:
            raise Exception(f"Error generating PDF report: {str(e)}")

    def generate_batch_pdf_report(self, results_df, marking_criteria=None):
        """Generate a PDF report for batch results."""
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=landscape(A4),
                rightMargin=2 * cm,
                leftMargin=2 * cm,
                topMargin=2 * cm,
                bottomMargin=2 * cm
            )

            styles = self._create_pdf_style()
            story = []

            # Add logo and title
            self._add_logo(story)
            story.append(Paragraph("Batch Assessment Report", styles['CustomTitle']))
            story.append(Spacer(1, 12))

            # Timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            story.append(Paragraph(f"Generated on: {timestamp}", styles['CustomBody']))
            story.append(Spacer(1, 20))

            # Summary Statistics
            story.append(Paragraph("Summary Statistics", styles['CustomHeading']))
            stats_data = [
                ["Total Submissions:", str(len(results_df))],
                ["Average Score:", f"{results_df['score'].mean():.2f}/20"],
                ["Median Score:", f"{results_df['score'].median():.2f}/20"],
                ["Standard Deviation:", f"{results_df['score'].std():.2f}"],
                ["Highest Score:", f"{results_df['score'].max():.2f}/20"],
                ["Lowest Score:", f"{results_df['score'].min():.2f}/20"]
            ]
            
            stats_table = Table(stats_data, colWidths=[200, 300])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0'))
            ]))
            story.append(stats_table)
            story.append(Spacer(1, 20))

            # Score Distribution Analysis
            story.append(Paragraph("Score Distribution Analysis", styles['CustomHeading']))
            
            # Calculate quartiles and add to report
            q1 = results_df['score'].quantile(0.25)
            q3 = results_df['score'].quantile(0.75)
            iqr = q3 - q1
            
            distribution_data = [
                ["First Quartile (Q1):", f"{q1:.2f}"],
                ["Median:", f"{results_df['score'].median():.2f}"],
                ["Third Quartile (Q3):", f"{q3:.2f}"],
                ["Interquartile Range:", f"{iqr:.2f}"]
            ]
            
            distribution_table = Table(distribution_data, colWidths=[200, 300])
            distribution_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8F9FA')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0'))
            ]))
            story.append(distribution_table)
            story.append(PageBreak())

            # Individual Results Table
            story.append(Paragraph("Individual Results", styles['CustomHeading']))
            
            # Sort results by score in descending order
            results_df_sorted = results_df.sort_values('score', ascending=False)
            
            table_data = [["Rank", "Student ID", "Filename", "Score", "Percentile"]]
            for idx, (_, row) in enumerate(results_df_sorted.iterrows(), 1):
                percentile = (len(results_df_sorted) - idx + 1) / len(results_df_sorted) * 100
                table_data.append([
                    str(idx),
                    row.get("student_id", "unknown"),
                    row["filename"],
                    f"{row['score']:.1f}/20",
                    f"{percentile:.1f}%"
                ])

            results_table = Table(table_data, colWidths=[50, 100, 300, 80, 80])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
            ]))
            story.append(results_table)

            # Build document and clean up
            doc.build(story)
            buffer.seek(0)
            
            # Force garbage collection to free memory
            story = None
            gc.collect()
            
            return buffer.getvalue()
            
        except Exception as e:
            raise Exception(f"Error generating batch PDF report: {str(e)}")

    def generate_csv_report(self, results_df: pd.DataFrame) -> str:
        """Generate a CSV report with student IDs and scores."""
        try:
            csv_data = results_df[['student_id', 'score', 'feedback']].copy()
            csv_data.columns = ['Student ID', 'Score', 'Feedback']
            return csv_data.to_csv(index=False)
        except Exception as e:
            raise Exception(f"Error generating CSV report: {str(e)}")
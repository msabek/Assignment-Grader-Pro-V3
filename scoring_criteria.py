class ScoringCriteria:
    def __init__(self):
        self.default_criteria = {
            'total_points': 20,
            'questions': {
                'question1': {
                    'total_points': 8,
                    'parts': {
                        'part_a': {
                            'total_points': 3,
                            'components': {
                                'fx_calculation': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct x-component calculation'
                                },
                                'fy_calculation': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct y-component calculation'
                                },
                                'fz_calculation': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct z-component calculation'
                                }
                            }
                        },
                        'part_b': {
                            'total_points': 1,
                            'components': {
                                'cartesian_vector': {
                                    'points': 1,
                                    'partial_credit': [0.5, 0.25],
                                    'criteria': 'Correct Cartesian vector form'
                                }
                            }
                        },
                        'part_c': {
                            'total_points': 3,
                            'components': {
                                'direction_angles': {
                                    'points': 2,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct calculation of angles α and β'
                                },
                                'angle_verification': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Verification of direction angle requirement'
                                }
                            }
                        },
                        'part_d': {
                            'total_points': 1,
                            'components': {
                                'magnitude_unit_vector': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct expression as magnitude and unit vector'
                                }
                            }
                        }
                    }
                },
                'question2': {
                    'total_points': 4,
                    'parts': {
                        'part_a': {
                            'total_points': 0.5,
                            'components': {
                                'force_magnitude': {
                                    'points': 0.5,
                                    'partial_credit': 0.25,
                                    'criteria': 'Correct calculation of force magnitude'
                                }
                            }
                        },
                        'part_b': {
                            'total_points': 0.5,
                            'components': {
                                'unit_vector': {
                                    'points': 0.5,
                                    'partial_credit': 0.25,
                                    'criteria': 'Correct calculation of unit vector'
                                }
                            }
                        },
                        'part_c': {
                            'total_points': 3,
                            'components': {
                                'position_vector': {
                                    'points': 0.5,
                                    'criteria': 'Correct formulation of position vector'
                                },
                                'unit_vector_ra': {
                                    'points': 0.5,
                                    'criteria': 'Correct formulation of unit vector'
                                },
                                'length_calculation': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct calculation of L'
                                },
                                'coordinates': {
                                    'points': 1,
                                    'criteria': 'Correct calculation of x and y coordinates'
                                }
                            }
                        }
                    }
                },
                'question3': {
                    'total_points': 8,
                    'parts': {
                        'part_a': {
                            'total_points': 3,
                            'components': {
                                'force_fa': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct expression of force FA'
                                },
                                'force_fb': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct expression of force FB'
                                },
                                'force_fc': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct expression of force FC'
                                }
                            }
                        },
                        'part_b': {
                            'total_points': 3,
                            'components': {
                                'resultant_force': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct calculation of resultant force'
                                },
                                'force_magnitude': {
                                    'points': 1,
                                    'partial_credit': 0.5,
                                    'criteria': 'Correct calculation of force magnitude'
                                },
                                'direction_angles': {
                                    'points': 1.5,
                                    'partial_credit': 0.25,
                                    'criteria': 'Correct calculation of direction angles'
                                }
                            }
                        },
                        'part_c': {
                            'total_points': 2,
                            'components': {
                                'angle_calculation': {
                                    'points': 2,
                                    'partial_credit': 1,
                                    'criteria': 'Correct calculation of angle θ'
                                }
                            }
                        }
                    }
                }
            },
            'grading_notes': {
                'calculation_emphasis': 'Emphasis on process and formula setup rather than specific numerical values',
                'rounding_tolerance': '5% tolerance applied to all calculations',
                'single_error_policy': 'Single critical error should not result in less than 75% of total points if procedures are correct',
                'partial_credit': 'Partial credit available for correct setup with arithmetic errors'
            }
        }
        self.current_criteria = self.default_criteria.copy()
    
    def update_criteria(self, new_criteria):
        """Update scoring criteria with new values."""
        if new_criteria:
            self.current_criteria.update(new_criteria)
    
    def reset_to_default(self):
        """Reset criteria to default values."""
        self.current_criteria = self.default_criteria.copy()
    
    def get_criteria(self):
        """Get current scoring criteria."""
        return self.current_criteria

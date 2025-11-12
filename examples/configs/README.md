# JABS-postprocess Example Configuration Files

This directory contains example configuration files for JABS-postprocess heuristic behavior classification, designed to help researchers understand and implement custom behavior detection algorithms.

## Quick Start

1. **Copy an example file** that matches your behavior of interest
2. **Modify the parameters** for your specific experimental conditions  
3. **Test with your data** using the command line interface
4. **Iterate and refine** based on validation results

```bash
# Example usage with the basic locomotion configuration
python -m jabs_postprocess.heuristic_classify \
    --project_folder /path/to/your/project \
    --behavior_config examples/configs/basic_locomotion.yaml \
    --feature_folder features
```

## Available Configuration Files

### `basic_locomotion.yaml` - **Beginner Level**
**Purpose**: Detect when a mouse is actively moving above a simple speed threshold.

**Key Features**:
- Single condition: centroid velocity > 2.0 cm/s
- Basic post-processing parameters
- Extensive comments explaining each section
- Perfect starting point for new users

**Use Cases**:
- Distinguishing movement from stationary periods
- Basic activity level quantification
- Locomotor response to treatments

**Learning Objectives**: 
- Understand basic YAML structure
- Learn core configuration parameters
- Practice simple threshold-based classification

---

### `complex_behavior.yaml` - **Advanced Level**  
**Purpose**: Detect "Active Center Exploration" using multiple simultaneous conditions.

**Key Features**:
- Multi-condition logic (ALL/ANY operators)
- Mathematical expressions (multiply, subtract, abs)
- Spatial reasoning (center zone detection)
- Behavioral state exclusion logic
- Demonstrates full system capabilities

**Use Cases**:
- Open field anxiety studies
- Spatial preference analysis
- Complex ethological phenotyping

**Learning Objectives**:
- Master nested logical operators
- Implement mathematical feature transformations
- Design multi-modal behavioral definitions
- Handle conflicting behavioral states

---

### `template_config.yaml` - **Reference Guide**
**Purpose**: Comprehensive reference showing ALL available options and syntax patterns.

**Key Features**:
- Complete operator reference (>, <, >=, <=, all, any, minimum, maximum)
- Mathematical expression examples (add, subtract, multiply, divide, abs)
- Feature path directory
- Parameter tuning guidelines
- Troubleshooting guide
- Performance optimization tips

**Use Cases**:
- Reference during configuration development
- Learning advanced syntax patterns
- Troubleshooting configuration issues
- Understanding system capabilities

**Learning Objectives**:
- Comprehensive system knowledge
- Advanced configuration techniques
- Best practices and optimization
- Debugging and validation strategies

## Choosing the Right Configuration

| **If you want to...** | **Start with** | **Complexity** |
|------------------------|----------------|----------------|
| Detect simple movement | `basic_locomotion.yaml` | * |
| Learn the basics | `basic_locomotion.yaml` | * |
| Detect complex behaviors | `complex_behavior.yaml` | **** |
| Understand all options | `template_config.yaml` | ***** |
| Create custom behaviors | `template_config.yaml` | ***** |

## Configuration Structure Overview

All configuration files follow the same basic structure:

```yaml
# 1. BEHAVIOR IDENTIFICATION
behavior: Your Behavior Name

# 2. POST-PROCESSING PARAMETERS  
interpolate: 5    # Fill small gaps
stitch: 10        # Merge nearby bouts
min_bout: 30      # Minimum episode duration

# 3. CLASSIFICATION LOGIC
definition:
  greater than:   # or other operators
    - feature_path feature_description
    - threshold_value
```

### Key Concepts

**Post-processing Parameters**:
- `interpolate`: Fills brief gaps in detection (handles tracking errors)
- `stitch`: Combines nearby behavioral bouts (creates continuous episodes)  
- `min_bout`: Sets minimum duration for valid behaviors (filters noise)

**Mathematical Operations**:
- `add`, `subtract`, `multiply`, `divide`: Basic arithmetic on features
- `abs`: Absolute value (useful for angular differences)

**Logical Operations**:  
- `all`: Every condition must be true (AND logic)
- `any`: At least one condition must be true (OR logic)
- `minimum`: At least N conditions must be true
- `maximum`: At most N conditions can be true

**Comparison Operations**:
- `greater than` (`>`, `gt`): Feature > threshold
- `less than` (`<`, `lt`): Feature < threshold  
- `greater than or equal` (`>=`, `gte`): Feature >= threshold
- `less than or equal` (`<=`, `lte`): Feature <= threshold

## Common Feature Paths

Here are the most frequently used features for behavior classification:

### Movement Features
```yaml
features/per_frame/centroid_velocity_mag          # Overall body speed
features/per_frame/point_speeds NOSE speed        # Head movement
features/per_frame/point_speeds BASE_NECK speed   # Neck movement
features/per_frame/point_speeds FRONT_PAW_L speed # Left paw movement
```

### Spatial Features  
```yaml
wall_distances/wall_0                             # Distance to wall 0
features/per_frame/corner_distances distance      # Distance to corner
avg_wall_length                                   # Arena dimension (constant)
```

### Orientation Features
```yaml
features/per_frame/head_angle head_angle          # Head direction
features/per_frame/wall_angle_0 wall_0_angle     # Angle relative to wall
```

## Parameter Tuning Guidelines

### Speed Thresholds (cm/second)
| Range | Typical Behaviors |
|-------|------------------|
| 0.5-2.0 | Micro-movements, grooming, sniffing |
| 2.0-5.0 | Walking, casual exploration |
| 5.0-15.0 | Active locomotion, investigation |
| 15.0-30.0 | Running, escape responses |
| >30.0 | Very rapid movement, panic |

### Temporal Thresholds (frames at 30 fps)
| Frames | Duration | Typical Use |
|--------|----------|-------------|
| 5-15 | 0.17-0.5s | Micro-behaviors, brief events |
| 15-30 | 0.5-1.0s | Short interactions |
| 30-90 | 1-3s | Moderate behavioral bouts |
| 90-300 | 3-10s | Extended activities |
| >300 | >10s | Sustained behavioral states |

### Distance Thresholds  
| Expression | Arena Fraction | Typical Use |
|------------|----------------|-------------|
| `avg_wall_length/10` | 10% | Very close to wall |
| `avg_wall_length/5` | 20% | Periphery zone |
| `avg_wall_length/3` | 33% | Center-periphery boundary |
| `avg_wall_length/2` | 50% | Arena center |

## Troubleshooting Guide

### Common Issues and Solutions

**"No behavior detected"**
- Lower detection thresholds
- Check that feature paths exist in your data
- Verify units match expected ranges

**"Too much behavior detected"** 
- Raise detection thresholds
- Add additional constraining conditions
- Increase minimum bout length

**"Fragmented behavior bouts"**
- Increase `stitch` parameter
- Decrease `min_bout` requirement
- Add interpolation for small gaps

**"YAML parsing errors"**
- Check indentation (use spaces, not tabs)
- Ensure proper nesting structure
- Validate syntax with online YAML checker

**"Feature not found errors"**
- Verify feature paths exist in your HDF5 files
- Check feature extraction completed successfully
- Confirm animal indices match your data

### Validation Workflow

1. **Start Simple**: Begin with single-condition definitions
2. **Visual Validation**: Check results against video footage
3. **Iterative Refinement**: Gradually add complexity and tune parameters
4. **Quantitative Validation**: Compare against manual scoring when possible
5. **Documentation**: Record parameter choices and biological rationale

## Scientific Best Practices

### Configuration Development
- **Document rationale**: Explain why you chose specific thresholds
- **Version control**: Track configuration changes alongside analysis code
- **Validate thoroughly**: Test with multiple animals and conditions
- **Share configurations**: Enable reproducibility by sharing validated configs

### Parameter Selection
- **Biologically motivated**: Base thresholds on known behavioral characteristics
- **Data-driven**: Use pilot data to inform reasonable parameter ranges
- **Context-appropriate**: Adjust for different experimental paradigms
- **Conservative first**: Start with stricter criteria, then relax as needed

### Quality Control
- **Manual verification**: Visually inspect classified bouts against video
- **Cross-validation**: Test configurations on independent datasets  
- **Sensitivity analysis**: Evaluate robustness to parameter changes
- **Peer review**: Have colleagues validate behavioral definitions

## Additional Resources

### JABS-postprocess Documentation
- [Main Repository](https://github.com/AdaptiveMotorControlLab/JABS-postprocess)
- [Feature Extraction Guide](../../src/jabs_postprocess/utils/features.py)
- [Heuristic Classification System](../../src/jabs_postprocess/utils/heuristics.py)

### Related Research
- **Open Field Analysis**: For spatial behavior paradigms
- **Behavioral Phenotyping**: For comprehensive ethological analysis
- **Motion Detection Algorithms**: For movement-based classifications

### Community
- Share your validated configurations
- Request specific behavioral examples
- Contribute improvements to example files

---

## Contributing

Found an issue or have a suggestion for improvement?

1. **Bug Reports**: File issues for incorrect examples or unclear documentation
2. **New Examples**: Contribute configurations for additional behavioral phenotypes
3. **Documentation**: Help improve clarity and completeness of comments
4. **Validation**: Share results from using these configurations with your data

---

*Happy behavior analysis!*
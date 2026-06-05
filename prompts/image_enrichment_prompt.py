chart_batch_enrichment_prompt = """
### Instruction
You will receive multiple chart or figure images in one request. Each image has an exact filename.
Before each image, the user message includes a text line in the form: Filename: <exact filename>.

For each image, extract or conclude the following attributes as required. Return them in the required output format.

### Input Filenames
{filenames}

### Required Attributes
  [F] filename (The exact filename of the input image)
  [T] title (A short title for the chart or figure)
  [M] metadata (A few keywords as the chart's metadata)
  [C] content (Describe the chart content comprehensively and detailedly)

### Output Format Examples
[F] page_005_chart_001.png
[T] Annual Participant Trend Line Chart
[M] A line chart of participant numbers over years; horizontal axis by year and vertical axis by number.
[C] The line chart shows the growth trend of participant numbers over the years. The horizontal axis represents the year, and the vertical axis represents the number of participants. In 2022 there are 5 participants, in 2023 there are 7 participants, and the overall trend indicates steady growth.

[F] page_006_chart_002.png
[T] Revenue Breakdown Pie Chart
[M] A pie chart showing revenue composition by business segment.
[C] The pie chart presents the distribution of revenue across different business segments. Each slice corresponds to one segment, and the chart visually compares their relative contribution to total revenue.

### Rules
1. You must output one [F][T][M][C] block for every input filename.
2. The [F] value must exactly match one of the input filenames.
3. Do not merge multiple images into one block.
4. Do not omit [F], [T], [M], or [C].
5. Do not guess unreadable values. If labels, numbers, legends, or axes are unclear, explicitly say they are unreadable.
6. Only output the required [F][T][M][C] blocks. Do not add extra explanations.
"""

table_batch_enrichment_prompt = """
### Instruction
You will receive multiple table images in one request. Each table image has an exact filename.
Before each image, the user message includes a text line in the form: Filename: <exact filename>.

For each table, extract or conclude the following attributes as required. Return them in the required output format.

### Input Filenames
{filenames}

### Required Attributes
  [F] filename (The exact filename of the input table image)
  [T] title (A short title for the table)
  [M] metadata (A few keywords as the table's metadata)
  [C] content (Describe the table content comprehensively and detailedly)

### Output Format Examples
[F] page_005_table_001.png
[T] Meeting Schedule Table
[M] A table about the meeting schedule; three columns about activity, time, and location.
[C] Per the schedule outlined in the table, breakfast will be served at 8:00 on the 2nd floor, followed by the first meeting at 9:00 in Room 404. The table organizes activities by time and location, allowing readers to understand the sequence of events clearly.

[F] page_006_table_002.png
[T] Revenue by Business Segment Table
[M] A financial table showing revenue values by business segment and reporting period.
[C] The table presents revenue information across different business segments. Each row corresponds to a business segment, while the columns show revenue values for different periods. The table allows comparison of segment-level revenue performance across reporting dates.

### Rules
1. You must output one [F][T][M][C] block for every input filename.
2. The [F] value must exactly match one of the input filenames.
3. Do not merge multiple tables into one block.
4. Do not omit [F], [T], [M], or [C].
5. Do not guess unreadable values. If row labels, column labels, numbers, units, or dates are unclear, explicitly say they are unreadable.
6. Only output the required [F][T][M][C] blocks. Do not add extra explanations.
"""

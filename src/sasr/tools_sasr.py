import base64
import os
import re
from pathlib import Path

# SIL reference examples configuration
RELEASE_ROOT = Path(__file__).resolve().parents[2]
SIL_EXAMPLE_DIR = str(RELEASE_ROOT / "assets" / "sil_examples")

SIL_EXAMPLES = [
	{
		"filename": "example_1.png",
		"category": "Vertical Arithmetic",
		"label": r"\begin{arith}\operand[0]{101}\operator{\times}\operand[0]{88}\result\operand[0]{808}\operand[1]{808}\result\operand[0]{8888}\end{arith}",
		"disambiguation": "Arithmetic context. No ambiguous symbols."
	},
	{
		"filename": "example_2.png",
		"category": "Long Division",
		"label": r"\begin{longdiv}\divisor{15}\dividend[1]{13.8}\quotient[0]{0.92}\operand[1]{13.5}\result\operand[0]{0.30}\operand[0]{0.30}\result\operand[0]{0}\end{longdiv}",
		"disambiguation": "Arithmetic context. Decimal points are clear."
	},
	{
		"filename": "example_3.png",
		"category": "Short Division",
		"label": r"\begin{shortdiv}\sdivstep[2]{1000&624}\sdivstep[2]{500&312}\sdivstep[2]{250&156}\sdivfinal{125&78}\end{shortdiv}",
		"disambiguation": "Arithmetic context. Numbers are clear."
	},
	{
		"filename": "example_4.png",
		"category": "Matrix",
		"label": r"$$=\begin{vmatrix}1&0&0\\0&\lambda-1&0\\0&0&(\lambda-1)^{2}\end{vmatrix}$$",
		"disambiguation": "Linear algebra context. \\lambda is a variable, distinct from numbers."
	},
	{
		"filename": "example_5.png",
		"category": "Equation System",
		"label": r"$$\begin{cases}2x+3y-z=2\textcircled{1}\\4x-2y+z=1\textcircled{2}\\3x-y-z=\frac{5}{2}\textcircled{3}\end{cases}$$",
		"disambiguation": "Algebraic context. x, y, z are variables. Circled numbers are labels."
	},
	{
		"filename": "example_6.jpg",
		"category": "Multi-line Equation",
		"label": r"$$\begin{aligned}&56\times67+56\times33\\=&56\times(67+33)\\=&56\times100\\=&5600\end{aligned}$$",
		"disambiguation": "Arithmetic context. \\times is multiplication, not variable x."
	},
	{
		"filename": "example_7.png",
		"category": "Chemical Equation",
		"label": r"$\ce{2H_{2}O +2SO_{2} +O_{2} ->2H_{2}SO_{4}}$",
		"disambiguation": "Chemistry context. O is Oxygen (element), not zero (0)."
	},
	{
        "filename": "example_8.jpg",
        "category": "Cancellation/Simplification",
        "label": r"$=\frac{3}{4}\times\frac{3}{\bcancel{4}1}\times\frac{\bcancel{4}1}{5}$",
        "disambiguation": "Arithmetic simplification. Crossed out numbers indicate cancellation."
    },
    {
        "filename": "example_9.png",
        "category": "Standard Formula with Artifacts",
        "label": r"$\therefore \frac{AB}{CF}=\frac{3}{2}$",
        "disambiguation": "Geometry context. A, B, C, F are points."
    }
]


def get_image_media_type(image_path: str) -> str:
	"""Determine the media type based on file extension.
	
	Args:
		image_path: Path to the image file
		
	Returns:
		Media type string like 'image/png', 'image/jpeg', etc.
	"""
	ext = os.path.splitext(image_path)[1].lower()
	media_types = {
		'.png': 'image/png',
		'.jpg': 'image/jpeg',
		'.jpeg': 'image/jpeg',
		'.gif': 'image/gif',
		'.bmp': 'image/bmp',
		'.webp': 'image/webp',
	}
	return media_types.get(ext, 'image/png')  # Default to PNG if extension not recognized


def encode_image(image_path: str) -> str:
	with open(image_path, "rb") as image_file:
		return base64.b64encode(image_file.read()).decode("utf-8")


def build_messages(class_name: str, image_path: str, mode: str = 'SASR'):
	"""Build the chat `messages` payload for a given class and image.

	- `class_name` is a paper category abbreviation such as 'SSE' or 'CHEM'.
	- `image_path` is the target image file to be processed.
	- `mode` is one of 'SIL', 'MSR', or 'SASR', matching the paper methods.

	Returns the list-of-messages structure expected by the OpenAI-compatible client.
	"""
	mode = mode.upper()
	if mode not in {'SIL', 'MSR', 'SASR'}:
		raise ValueError(f"Unsupported mode: {mode}")

	system_message = (
		r"You are a LaTeX transcription expert.\n"
		r"CRITICAL: You must use the following CUSTOM SYNTAX for specific arithmetic structures:\n"
		r"1. Vertical arithmetic (+,-,\times): \begin{arith}...\end{arith}, each command auto-breaks line. \operand[indent]{num} ([indent]=right-align level, 0=none), \operator{symbol} (+,-,\times), \result (horizontal line). Ex: \begin{arith}\operand[0]{123}\operator{+}\operand[0]{456}\result\operand[0]{579}\end{arith}\n"
		r"2. Long division: \begin{longdiv}...\end{longdiv}, each command auto-breaks line. \divisor{num}, \dividend[indent]{num} (top), \quotient[indent]{num} (result above), \operand[indent]{num} (steps), \result (line). Ex: \begin{longdiv}\divisor{15}\dividend[1]{138}\quotient[0]{9.2}\operand[1]{135}\result\operand[0]{30}\operand[0]{30}\result\operand[0]{0}\end{longdiv}\n"
		r"3. Short division: \begin{shortdiv}...\end{shortdiv}, each command auto-breaks line. \sdivstep[divisor]{n1&n2&...} ([divisor]=divisor num, & aligns columns), \sdivfinal{n1&n2&...} (final row). Ex: \begin{shortdiv}\sdivstep[2]{12&18}\sdivstep[3]{6&9}\sdivfinal{2&3}\end{shortdiv}\n"
		r"4. Matrices: \begin{bmatrix}/\begin{pmatrix}/\begin{vmatrix}\n"
		r"5. Equation systems: \begin{cases}\n"
		r"6. Multi-line equations: \begin{aligned}\n"
		r"7. Chemical equations: \ce{...}\n\n"
		
		r"CRITICAL RULES" + "\n"
		r"Do NOT add tokens, symbols, or formatting not present in the image" + "\n"
		r"Transcribe ONLY what is visible, maintaining exact visual appearance" + "\n"
		r"Transcribe EXACTLY as written in the image. Because the writing might be incorrect, do NOT convert it to standard content on your own." + "\n"
	)
	
	if mode == 'MSR':
		system_message += "\n\nHere are some examples of categories and their corresponding LaTeX labels (for reference only, no images provided):\n"
		for example in SIL_EXAMPLES:
			system_message += f"Category: {example['category']}\nLabel: {example['label']}\n\n"
	
	# User message with Chain of Thought instructions
	msr_prompt_text = (
		r"Please transcribe this handwritten mathematical image using the following Chain of Thought process:" + "\n\n"
		
		r"Step 1: Formula Type Classification" + "\n"
		r"Carefully analyze the image and classify the formula type into ONE of these categories:" + "\n"
		r"1. Vertical Arithmetic" + "\n"
		r"2. Long Division" + "\n"
		r"3. Short Division" + "\n"
		r"4. Matrix" + "\n"
		r"5. Equation System" + "\n"
		r"6. Multi-line Equation" + "\n"
		r"7. Chemical Equation" + "\n"
		r"8. Cancellation/Simplification" + "\n\n"
		r"9. Standard Formula with Artifacts" + "\n\n"
		r"10. Single-line formula" + "\n\n"
		r"Output your classification as: <CLASSIFICATION>category_name</CLASSIFICATION>" + "\n\n"
		
		r"Step 2: Initial Transcription" + "\n"
		r"Transcribe the formula using appropriate LaTeX syntax based on your classification." + "\n"
		r"Output as: <TRANSCRIPTION>your_latex_code</TRANSCRIPTION>" + "\n\n"
		
		r"Step 3: Semantic Disambiguation" + "\n"
		r"Use context to disambiguate symbols. For example:" + "\n"
		r"In velocity formulas, 'v' is usually velocity, not 'V'." + "\n"
		r"In circular formulas, '\omega' is angular velocity, not 'w'." + "\n"
		r"Output result as: <DISAMBIGUATION>explanation</DISAMBIGUATION>" + "\n\n"
		
		r"Step 4: Final Output" + "\n"
		r"Based on the disambiguation, output the final corrected LaTeX enclosed in \boxed{}." + "\n"
		r"Do NOT output markdown code blocks (no ```latex)." + "\n"
		r"Output only: \boxed{your_final_latex_here}"
	)

	simple_prompt_text = "Please transcribe this handwritten mathematical image."

	# Build messages list starting with system message
	messages = [
		{"role": "system", "content": system_message},
	]
	
	# Add SIL reference examples
	if mode in ['SIL', 'SASR']:
		for example in SIL_EXAMPLES:
			example_path = os.path.join(SIL_EXAMPLE_DIR, example["filename"])
			example_media_type = get_image_media_type(example_path)
			
			text = "Please transcribe this handwritten mathematical image."
			if mode == 'SIL':
				text = "Please identify the category of this image."
			# User message with example image
			example_user_content = [
				{"type": "text", "text": text},
				{
					"type": "image_url",
					"image_url": {"url": f"data:{example_media_type};base64,{encode_image(example_path)}"},
				}
			]
			messages.append({"role": "user", "content": example_user_content})
			
			# Assistant response
			if mode == 'SASR':
				example_response = (
					f"<CLASSIFICATION>{example['category']}</CLASSIFICATION>\n"
					f"<TRANSCRIPTION>{example['label']}</TRANSCRIPTION>\n"
					f"<DISAMBIGUATION>{example.get('disambiguation', 'pass')}</DISAMBIGUATION>\n"
					f"\\boxed{{{example['label']}}}"
				)
			else: # SIL only
				example_response = f"<CLASSIFICATION>{example['category']}</CLASSIFICATION>"

			messages.append({"role": "assistant", "content": example_response})
	
	# Add the actual user query with the target image
	media_type = get_image_media_type(image_path)
	
	if mode in ['MSR', 'SASR']:
		final_prompt = msr_prompt_text
	else:
		final_prompt = simple_prompt_text

	content_list = [
		{"type": "text", "text": final_prompt},
		{
			"type": "image_url",
			"image_url": {"url": f"data:{media_type};base64,{encode_image(image_path)}"},
		}
	]
	messages.append({"role": "user", "content": content_list})
	
	return messages

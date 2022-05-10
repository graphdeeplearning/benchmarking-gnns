
# Clean the main.py file after conversion from notebook.
# Any notebook code is removed from the main.py file.


import subprocess


def cleaner_main(filename):

	# file names
	file_notebook = filename + '.ipynb'
	file_python = filename + '.py'


	# convert notebook to python file
	print('Convert ' + file_notebook + ' to ' + file_python)
	subprocess.check_output('jupyter nbconvert --to script ' + str(file_notebook) , shell=True)

	print('Clean ' + file_python)

	# open file
	with open(file_python, "r") as f_in:
	    lines_in = f_in.readlines()

	# remove cell indices
	lines_in = [ line for i,line in enumerate(lines_in) if '# In[' not in line ]

	# remove comments
	lines_in = [ line for i,line in enumerate(lines_in) if line[0]!='#' ]

	# remove "in_ipynb()" function
	idx_start_fnc = next((i for i, x in enumerate(lines_in) if 'def in_ipynb' in x), None)
	if idx_start_fnc!=None:
	    idx_end_fnc = idx_start_fnc + next((i for i, x in enumerate(lines_in[idx_start_fnc+1:]) if x[:4] not in ['\n','    ']), None)  
	    lines_in = [ line for i,line in enumerate(lines_in) if i not in range(idx_start_fnc,idx_end_fnc+1) ]
	list_elements_to_remove = ['in_ipynb()', 'print(notebook_mode)']
	for elem in list_elements_to_remove:
	    lines_in = [ line for i,line in enumerate(lines_in) if elem not in line ]
	    
	# unindent "if notebook_mode==False" block
	idx_start_fnc = next((i for i, x in enumerate(lines_in) if 'if notebook_mode==False' in x), None)
	if idx_start_fnc!=None:
	    idx_end_fnc = idx_start_fnc + next((i for i, x in enumerate(lines_in[idx_start_fnc+1:]) if x[:8] not in ['\n','        ']), None)
	    for i in range(idx_start_fnc,idx_end_fnc+1):
	        lines_in[i] = lines_in[i][4:]
	    lines_in.pop(idx_start_fnc)
	list_elements_to_remove = ['# notebook mode', '# terminal mode']
	for elem in list_elements_to_remove:
	    lines_in = [ line for i,line in enumerate(lines_in) if elem not in line ]

	# remove remaining "if notebook_mode==True" blocks - single indent
	run = True
	while run:
	    idx_start_fnc = next((i for i, x in enumerate(lines_in) if x[:16]=='if notebook_mode'), None)
	    if idx_start_fnc!=None:
	        idx_end_fnc = idx_start_fnc + next((i for i, x in enumerate(lines_in[idx_start_fnc+1:]) if x[:4] not in ['\n','    ']), None)  
	        lines_in = [ line for i,line in enumerate(lines_in) if i not in range(idx_start_fnc,idx_end_fnc+1) ]
	    else:
	        run = False
       
	# remove "if notebook_mode==True" block - double indents
	idx_start_fnc = next((i for i, x in enumerate(lines_in) if x[:20]=='    if notebook_mode'), None)
	if idx_start_fnc!=None:
		idx_end_fnc = idx_start_fnc + next((i for i, x in enumerate(lines_in[idx_start_fnc+1:]) if x[:8] not in ['\n','        ']), None)  
		lines_in = [ line for i,line in enumerate(lines_in) if i not in range(idx_start_fnc,idx_end_fnc+1) ]

	# prepare main() for terminal mode
	idx = next((i for i, x in enumerate(lines_in) if 'def main' in x), None)
	if idx!=None: lines_in[idx] = 'def main():'
	idx = next((i for i, x in enumerate(lines_in) if x[:5]=='else:'), None)
	if idx!=None: lines_in.pop(idx)
	idx = next((i for i, x in enumerate(lines_in) if x[:10]=='    main()'), None)
	if idx!=None: lines_in[idx] = 'main()'

	# remove notebook variables
	idx = next((i for i, x in enumerate(lines_in) if 'use_gpu = True' in x), None)
	if idx!=None: lines_in.pop(idx)
	idx = next((i for i, x in enumerate(lines_in) if 'gpu_id = -1' in x), None)
	if idx!=None: lines_in.pop(idx)
	idx = next((i for i, x in enumerate(lines_in) if 'device = None' in x), None)
	if idx!=None: lines_in.pop(idx)
	run = True
	while run:
		idx = next((i for i, x in enumerate(lines_in) if x[:10]=='MODEL_NAME'), None)
		if idx!=None: 
			lines_in.pop(idx)
		else:
			run = False

	# save clean file
	lines_out = str()
	for line in lines_in: lines_out += line
	with open(file_python, 'w') as f_out:
	    f_out.write(lines_out)
	    
	print('Done. ')






import requests
import pandas as pd

def main():
	f = open('numbers.txt', 'r+')
	i = 0
	lst = []
	for line in f:
		dic = {'TailNumber': line[:-1]}
		r = requests.get('http://registry.faa.gov/aircraftinquiry/NNum_Results.aspx?NNumbertxt={}'.format(line))
		s = r.text

		# Add len(' class="Results_DataText\">') = 26
		start = s.find('lbMfrName\"') + len('lbMfrName\"') + 26
		end = 0
		while (s[start + end] != ' '):
			end += 1
		if s[start:start+end] == 'D':
			dic.update({'Manu': 0, 'Model': 0, 'CertificateDate': 0})
			lst.append(dic)
			continue
		dic['Manu'] = s[start:start + end]

		start = s.find('content_Label7\"') + len('content_Label7\"') + 26
		end = 0
		while (s[start + end] != ' '):
			end += 1
		dic['Model'] = s[start:start + end]

		start = s.find('lbCertDate\"') + len('lbCertDate\"') + 26
		end = 0
		while (s[start + end] != '<'):
			end += 1
		dic['CertificateDate'] = s[start:start + end]

		lst.append(dic)
		i += 1

	f.close()
	df = pd.DataFrame(lst)
	df.to_csv('plane_data3.csv')

if __name__ == '__main__':
	main()
# parser.py (Corrected)

from bs4 import BeautifulSoup
import json
import zipfile
import io

def _parse_format_A(soup):
    """Parses our original mock XML format."""
    print("Detected Format A.")
    structured_data = []
    
    def find_text(tag, name):
        found = tag.find(name)
        return found.text.strip() if found else "Unknown"

    for msg in soup.find_all('Message'):
        from_party = "Unknown"
        for party in msg.find_all('Party'):
            if party.get('role') == 'From':
                from_party = party.text.strip()
        structured_data.append({
            'type': 'chat', 'timestamp': find_text(msg, 'TimeStamp'),
            'sender': from_party, 'content': find_text(msg, 'Body')
        })

    for call in soup.find_all('Call'):
        from_party = "Unknown"
        for party in call.find_all('Party'):
            if party.get('role') == 'From': from_party = party.text.strip()
        structured_data.append({
            'type': 'call', 'timestamp': find_text(call, 'TimeStamp'),
            'direction': find_text(call, 'Direction'), 'number_or_contact': from_party
        })

    for contact in soup.find_all('Contact'):
        structured_data.append({
            'type': 'contact', 'name': find_text(contact, 'Name'),
            'number': find_text(contact, 'Phone')
        })
    return structured_data

def _parse_format_B(soup):
    """Parses the new mock XML format B."""
    print("Detected Format B.")
    structured_data = []

    def find_text(tag, name):
        found = tag.find(name)
        return found.text.strip() if found else "Unknown"

    for msg in soup.find_all('sms'):
        structured_data.append({
            'type': 'chat', 'timestamp': find_text(msg, 'timestamp'),
            'sender': find_text(msg, 'sender') if find_text(msg, 'direction') == 'incoming' else 'Device Owner',
            'content': find_text(msg, 'body')
        })

    for call in soup.find_all('call_record'):
        structured_data.append({
            'type': 'call', 'timestamp': find_text(call, 'date'),
            'direction': find_text(call, 'type'),
            'number_or_contact': find_text(call, 'number')
        })
        
    # --- THIS IS THE FIX ---
    return structured_data

def detect_format(soup):
    """Detects the XML format by looking for unique tags."""
    if soup.find('Chats'):
        return 'A'
    elif soup.find('sms_messages'):
        return 'B'
    else:
        return None

def parse_ufdr(file_path):
    """
    Main parsing function. Unzips, detects format, and dispatches to the correct parser.
    """
    xml_content = None
    with zipfile.ZipFile(file_path, 'r') as zf:
        for filename in zf.namelist():
            if filename.lower().endswith('.xml'):
                xml_content = zf.read(filename)
                break
    
    if not xml_content:
        raise ValueError("No XML file found in the UFDR ZIP archive.")

    soup = BeautifulSoup(xml_content, 'xml')
    
    format_type = detect_format(soup)
    
    # --- ADDED A DEBUG PRINT ---
    print(f"--- PARSER DEBUG: Detected format is: {format_type} ---")
    
    if format_type == 'A':
        return _parse_format_A(soup)
    elif format_type == 'B':
        return _parse_format_B(soup)
    else:
        raise ValueError("Unknown or unsupported XML report format.")
params = {
        "modelId": "6ef27501-b0ea-46d6-911a-cddb5aa76936",
        "version": "V1",
        "guid": "624cfcc044bc4111aa535c62e13ac5ad"
    }

url = "https://testhb.turingtopia.com/datasience/xquery/dsModel/downloadModelForModelService"

import requests
result = requests.get(url,params)


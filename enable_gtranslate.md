If you don't have a gcloud account with credit card yet: 
* go to https://console.cloud.google.com/?pli=1, akzeptiere nutzungsbedingungen. 
* gehe zu https://console.cloud.google.com/billing , richte dort ein privatkonto ein. 
* Create new Project: https://console.cloud.google.com/projectcreate?previousPage=%2Fcloud-resource-manager%3Fhl%3Dde%26project%3D%26folder%3D%26organizationId%3D&hl=de

In any case:
* Enable Translate API for it: https://console.cloud.google.com/flows/enableapi?apiid=translate.googleapis.com&hl=de
* Folge dem einen Schritt bei https://cloud.google.com/translate/docs/setup?hl=de#creating_service_accounts_and_keys (nur den einen!)
   * https://console.cloud.google.com/projectselector2/iam-admin/serviceaccounts?hl=de&supportedpurview=project
   * "Dienstkonto erstellen". Zugriff: -> "Schnellzugriff" -> "Einfach" -> "Inhaber"
   * drauf klicken -> "Schlüssel" -> "Schlüssel hinzufügen" -> "Schllüssel erstellen" -> "Json" finally done.
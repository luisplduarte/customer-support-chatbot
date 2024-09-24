export function combineDocuments(docs){
    console.log("docs = docs")
    return docs.map((doc)=>doc.pageContent).join('\n\n')
}
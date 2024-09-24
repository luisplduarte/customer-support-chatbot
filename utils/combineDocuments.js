export default function combineDocuments(docs){
    return docs.flat().map((doc) => doc.pageContent).join('\n\n\n\n')
}
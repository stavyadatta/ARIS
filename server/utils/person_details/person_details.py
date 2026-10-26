class PersonDetails:
    def __init__(self, person_dict: dict={}) -> None:
        self.person_dict = person_dict
        self.image = None
        self.latest_usr_msg = {}
        self.latest_llm_msg = {}

        self.all_relevant_messages = []

    def __bool__(self):
        return bool(self.person_dict)

    def get_person(self):
        if self.person_dict:
            return self.person_dict
        else:
            raise Exception("The person object is empty")

    def get_attribute(self, field: str):
        return self.person_dict.get(field, [])

    def get_latest_user_message(self):
        return self.latest_usr_msg

    def get_latest_llm_message(self):
        return self.latest_llm_msg

    def set_latest_usr_message(self, text):
        self.latest_usr_msg = text

    def set_latest_llm_message(self, llm_msg: dict):
        """
            Sets llm message to the llm variable to be used by Neo4j 
            database when adding messages to the linked list
        """
        self.latest_llm_msg = llm_msg

    def set_relevant_messages(self, msg_lst):
        self.all_relevant_messages = msg_lst

    def get_relevant_messages(self):
        return self.all_relevant_messages

    def set_attribute(self, update_field: str, update_string: str):
        """
            The function updates the person details within the object

            :param update_field: The field or the key that needs to be updated
            :param update_string: The string that will be used to update the dictionary
        """
        if self.person_dict:
            self.person_dict[update_field] = update_string
        else:
            raise Exception("The person object is empty")


    def set_image(self, image):
        self.image = image
